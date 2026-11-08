"""Small result objects; data frames for summaries and selected labelled series."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import storage as st
from .config import CostAssumptions, Scenario
from .data import open_prepared


class Run:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.meta = st.read_json(self.path / "run.json")
        if self.meta["format"] != st.FORMAT:
            raise ValueError("Unsupported run format; use the documented data importer")
        self.project = Path(self.meta["project"])
        self.scenario = Scenario(**self.meta["scenario"])
        self.operation_path = Path(self.meta["operations"])
        self.flow_path = Path(self.meta["flow"])
        self.layout_path = Path(self.meta["layout"])

    @property
    def name(self):
        return self.meta["name"]

    @property
    def id(self):
        return self.meta["id"]

    def _data(self):
        return open_prepared(self.meta["prepared"])

    def _layout(self):
        return st.load_arrays(self.layout_path / "layout.npz")

    @cached_property
    def nodes(self):
        return pd.read_parquet(self.layout_path / "nodes.parquet")

    @cached_property
    def links(self):
        return pd.read_parquet(self.flow_path / "links.parquet")

    @property
    def assumptions(self):
        prepared = st.read_json(Path(self.meta["prepared"]) / "prepared.json")
        archive = st.read_json(Path(prepared["archive"]) / "archive.json")
        return {
            **self.scenario.describe(),
            "weather": archive["weather"],
            "start": prepared["start"],
            "stop": prepared["stop"],
            "selected_alpha": self.alpha.to_dict(),
            "selected_length_exponent": st.read_json(self.flow_path / "complete.json")["exponent"],
        }

    @property
    def diagnostics(self):
        """Optimization trials and numerical conservation evidence, from manifests."""
        return {
            "placement": st.read_json(self.layout_path / "complete.json")["diagnostics"],
            "network": st.read_json(self.flow_path / "complete.json"),
        }

    @property
    def alpha(self):
        return pd.Series(
            st.read_json(self.layout_path / "complete.json")["alpha"], name="wind_share"
        )

    @property
    def layout(self):
        capacity = self._layout()["installed"]
        totals = st.load_arrays(self.operation_path / "totals.npz")
        return self.nodes.assign(
            wind_mw=capacity[0], solar_mw=capacity[1], backup_mw=totals["backup_capacity"]
        )

    @property
    def annual(self):
        return pd.read_parquet(self.operation_path / "annual.parquet")

    @property
    def summary(self):
        energy = st.load_arrays(self.operation_path / "totals.npz")
        flow = st.read_json(self.flow_path / "complete.json")
        return pd.DataFrame(
            [
                {
                    "run": self.name,
                    "preset": self.scenario.preset,
                    "nodes": len(self.nodes),
                    "links": len(self.links),
                    "demand_twh": energy["load"].sum() / 1e6,
                    "backup_twh": energy["backup"].sum() / 1e6,
                    "curtailment_twh": energy["curtailment"].sum() / 1e6,
                    "transmission_mw_km": float(self.links.capacity_mw @ self.links.length_km),
                    "max_balance_residual_mw": flow["max_residual_mw"],
                }
            ]
        )

    def series(self, variable, *, start=None, stop=None, nodes=None, links=None):
        """Read a selected interval. stop is exclusive. Power units are MW."""
        meta = st.read_json(Path(self.meta["prepared"]) / "prepared.json")
        begin = pd.Timestamp(meta["start"])
        end = pd.Timestamp(meta["stop"])

        def utc(value):
            t = pd.Timestamp(value)
            return t.tz_convert("UTC").tz_localize(None) if t.tz is not None else t

        a, b = utc(start) if start is not None else begin, utc(stop) if stop is not None else end
        if a < begin or b > end or b <= a or a != a.floor("h") or b != b.floor("h"):
            raise ValueError("Requested series must lie within the run on whole UTC hours")
        first = int((a - begin) / pd.Timedelta(hours=1))
        last = int((b - begin) / pd.Timedelta(hours=1))
        selection = slice(first, last)
        time = pd.date_range(a, b, freq="h", inclusive="left")
        if variable == "flow":
            axis = "link"
            ids = self.links.link_id.to_numpy()
            chosen = links
        else:
            axis = "node"
            ids = self.nodes.node_id.to_numpy()
            chosen = nodes
        positions = (
            np.arange(len(ids))
            if chosen is None
            else pd.Index(ids).get_indexer(np.atleast_1d(chosen))
        )
        if (positions < 0).any():
            raise KeyError("Unknown node or link in series selection")
        required = (last - first) * len(positions) * 8 * (2 if variable == "generation" else 1)
        if required > self.meta["memory_gib"] * 1024**3 * 0.5:
            raise MemoryError(
                "Requested series is too large; select a smaller interval or node/link set"
            )
        op = st.group(self.operation_path / "arrays.zarr")
        if variable == "flow":
            values = st.group(self.flow_path / "arrays.zarr")["flow"].oindex[selection, positions]
        elif variable in ("generation", "balancing"):
            values = op[variable].oindex[selection, positions]
        elif variable in ("load", "backup", "curtailment", "injection"):
            if variable in ("load", "injection"):
                data = self._data()
                from .data import expand_load

                load = np.zeros((last - first, len(positions)))
                land = positions < len(data.nodes)
                load[:, land] = expand_load(data.template.iloc[:, positions[land]], time)
            if variable == "load":
                values = load
            else:
                balance = op["balancing"].oindex[selection, positions]
                values = np.maximum(-balance, 0) if variable == "backup" else np.maximum(balance, 0)
                if variable == "injection":
                    values = (
                        op["generation"].oindex[selection, positions].sum(axis=2) - load - balance
                    )
        else:
            raise ValueError(
                "Unknown series; choose generation, balancing, load, backup, "
                "curtailment, injection or flow"
            )
        dims = ("time", axis, "renewable") if variable == "generation" else ("time", axis)
        coords = {"time": time, axis: ids[positions]}
        if variable == "generation":
            coords["renewable"] = ["wind", "solar"]
        return xr.DataArray(values, dims=dims, coords=coords, name=variable, attrs={"units": "MW"})

    def evaluate(
        self,
        *,
        method="flow_traced",
        split="minimize_wsd",
        bins=50,
        costs=None,
        cost_table=None,
        progress=True,
    ):
        from .allocation import trace
        from .economics import Prices, evaluate_costs

        if split != "minimize_wsd" and (not isinstance(split, (int, float)) or not 0 <= split <= 1):
            raise ValueError("split must be in [0,1] or 'minimize_wsd'")
        assumptions = costs or CostAssumptions()
        if not isinstance(assumptions, CostAssumptions):
            raise TypeError("costs must be CostAssumptions; use cost_table for a CSV path")
        source = Path(st.read_json(Path(self.meta["prepared"]) / "prepared.json")["source"])
        table_path = Path(cost_table) if cost_table is not None else source / "costs_2030.csv"
        if cost_table is not None and not table_path.is_file():
            raise FileNotFoundError(f"Requested cost table does not exist: {table_path}")
        if split == "minimize_wsd" and not table_path.is_file():
            raise ValueError(
                "minimize_wsd requires costs; supply cost_table or "
                "select a fixed split for capacity-only results"
            )
        installed = np.vstack(
            (
                self._layout()["installed"],
                st.load_arrays(self.operation_path / "totals.npz")["backup_capacity"],
            )
        )
        if table_path.is_file():
            p = Prices(pd.read_csv(table_path), assumptions)
            from .economics import validate_prices

            validate_prices(p, self.nodes, self.links, installed)
        allocation_path = trace(self, method, bins, progress=progress)
        definition = {
            "allocation": allocation_path.name,
            "costs": st.file_hash(table_path) if table_path.is_file() else None,
            "assumptions": st.plain(assumptions),
            "split": split,
            "algorithm": 4,
        }
        path = self.project / ".eurotracer" / "evaluation" / st.identity(definition)
        with st.lock(path):
            if not (path / "evaluation.json").exists():
                a = st.load_arrays(allocation_path / "allocation.npz")
                summary = {"split": float(split) if split != "minimize_wsd" else 0.5}
                if table_path.is_file():
                    table = pd.read_csv(table_path)
                    table.to_csv(path / "costs.csv", index=False)
                    p = Prices(table, assumptions)
                    installed = np.vstack(
                        (
                            self._layout()["installed"],
                            st.load_arrays(self.operation_path / "totals.npz")["backup_capacity"],
                        )
                    )
                    hours = st.group(self.operation_path / "arrays.zarr")["balancing"].shape[0]
                    nt, ct, ni, ci, summary = evaluate_costs(
                        self.nodes, self.links, installed, a, hours, p, split
                    )
                    for name, frame in (
                        ("nodes", nt),
                        ("countries", ct),
                        ("installed_nodes", ni),
                        ("installed_countries", ci),
                    ):
                        frame.to_parquet(path / f"{name}.parquet", index=False)
                from .economics import capacity_tables

                for suffix, physical in (("", False), ("installed_", True)):
                    nt, ct = capacity_tables(
                        self.nodes, self.links, installed, a, summary["split"], physical=physical
                    )
                    for name, frame in ((suffix + "nodes", nt), (suffix + "countries", ct)):
                        file = path / f"{name}.parquet"
                        if table_path.is_file():
                            keys = ["node_id", "country"] if name.endswith("nodes") else ["country"]
                            frame = frame.merge(
                                pd.read_parquet(file), on=keys, validate="one_to_one"
                            )
                        frame.to_parquet(file, index=False)
                st.write_json(
                    path / "evaluation.json",
                    {
                        **definition,
                        "run": str(self.path),
                        "method": method,
                        "allocation_path": str(allocation_path),
                        "has_costs": table_path.is_file(),
                        "summary": summary,
                    },
                )
        return Evaluation(path, run=self)

    def plot(self, kind="layout", **kwargs):
        from .plotting import plot

        return plot(self, kind, **kwargs)


class Evaluation:
    def __init__(self, path, run=None):
        self.path = Path(path)
        self.meta = st.read_json(self.path / "evaluation.json")
        self.run = run or Run(self.meta["run"])

    @property
    def summary(self):
        return pd.Series(
            {"run": self.run.name, "method": self.meta["method"], **self.meta["summary"]}
        )

    @property
    def diagnostics(self):
        return st.read_json(Path(self.meta["allocation_path"]) / "complete.json")

    @property
    def assumptions(self):
        return {
            "method": self.meta["method"],
            "split": self.meta["summary"]["split"],
            "bins": self.diagnostics["bins"],
            "costs": self.meta["assumptions"],
            "cost_table": str(self.path / "costs.csv") if self.meta["has_costs"] else None,
        }

    def _table(self, name):
        return pd.read_parquet(self.path / f"{name}.parquet")

    @property
    def countries(self):
        return self._table("countries")

    @property
    def nodes(self):
        return self._table("nodes")

    @property
    def installed_countries(self):
        return self._table("installed_countries")

    @property
    def installed_nodes(self):
        return self._table("installed_nodes")

    @property
    def arrays(self):
        a = st.load_arrays(Path(self.meta["allocation_path"]) / "allocation.npz")
        ids = self.run.nodes.node_id.to_numpy()
        links = self.run.links.link_id.to_numpy()
        split = self.meta["summary"]["split"]
        return xr.Dataset(
            {
                "generation_capacity": (
                    ("technology", "source_node", "sink_node"),
                    a["generation"],
                    {"units": "MW"},
                ),
                "average_export": (
                    ("technology", "source_node", "sink_node"),
                    a["exports"],
                    {"units": "MW"},
                ),
                "backup_energy": (
                    ("source_node", "sink_node"),
                    a["backup_energy"],
                    {"units": "MWh"},
                ),
                "link_capacity": (
                    ("sink_node", "link"),
                    (1 - split) * a["reverse"] + split * a["forward"],
                    {"units": "MW"},
                ),
            },
            coords={
                "technology": ["wind", "solar", "backup"],
                "source_node": ids,
                "sink_node": ids,
                "link": links,
            },
        )

    def plot(self, kind="costs", **kwargs):
        from .plotting import plot

        return plot(self, kind, **kwargs)


def compare(*evaluations):
    """Country comparisons against the first evaluation; penalties in attrs."""
    if not evaluations:
        raise ValueError("Supply at least one evaluation")
    if any(not e.meta["has_costs"] for e in evaluations):
        raise ValueError("LCOE comparison requires valid costs for every evaluation")
    reference = evaluations[0].countries.set_index("country")
    frames = []
    metrics = []
    for i, evaluation in enumerate(evaluations):
        if not evaluation.meta["has_costs"]:
            raise ValueError("LCOE comparison requires valid costs for every evaluation")
        frame = evaluation.countries.set_index("country")
        if set(frame.index) != set(reference.index):
            raise ValueError("Compared evaluations must contain the same countries")
        frame = frame.reindex(reference.index)
        delta = frame.total_eur_per_mwh - reference.total_eur_per_mwh
        weights = reference.load_mwh_per_year / reference.load_mwh_per_year.sum()
        penalty = float(weights @ (np.maximum(delta.fillna(0), 0) ** 2))
        frames.append(
            frame.assign(
                comparison=i,
                run=evaluation.run.name,
                method=evaluation.meta["method"],
                split=evaluation.meta["summary"]["split"],
                delta_eur_per_mwh=delta,
            ).reset_index()
        )
        metrics.append(
            {
                "comparison": i,
                "positive_part_squared_penalty": penalty,
                "all_countries_no_worse": bool((delta.dropna() <= 1e-9).all()),
            }
        )
    result = pd.concat(frames, ignore_index=True)
    result.attrs["participation"] = metrics
    return result
