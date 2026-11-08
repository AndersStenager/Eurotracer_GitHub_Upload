"""Blockwise generation, synchronized balancing, and exact network sizing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from . import placement
from . import storage as st
from .network import Network, balancing, build_links


def operate(data, scenario, layout_path, *, memory_gib=24, progress=True):
    layout = st.load_arrays(layout_path / "layout.npz")
    nodes = pd.read_parquet(layout_path / "nodes.parquet")
    meta = st.read_json(layout_path / "complete.json")
    definition = {
        "layout": layout_path.name,
        "balancing": scenario.balancing,
        "quantile": scenario.capacity_quantile,
        "algorithm": 3,
    }
    path = data.project / ".eurotracer" / "operations" / st.identity(definition)
    with st.lock(path):
        if (path / "complete.json").exists():
            if progress:
                print("Reusing physical operation")
            return path
        if progress:
            print("Calculating generation and synchronized balancing")
        root = st.group(path / "arrays.zarr", "a")
        shape = (len(data.time), len(nodes))
        generation = st.array(root, "generation", (*shape, 2))
        balance = st.array(root, "balancing", shape)
        cf = st.group(Path(meta["basis_path"]) / "arrays.zarr")
        groups = (
            pd.factorize(nodes.country, sort=True)[0]
            if scenario.balancing == "country"
            else np.zeros(len(nodes), int)
        )
        checkpoint = path / "progress.json"
        done = st.read_json(checkpoint)["hours"] if checkpoint.exists() else 0
        last_year = None
        for block in st.slices(len(data.time)):
            if block.stop <= done:
                continue
            g = placement.generation_block(data, scenario, layout, block, cf)
            load = np.pad(data.load(block), ((0, 0), (0, len(nodes) - len(data.nodes))))
            b, _ = balancing(g, load, groups, layout["mean_load"])
            generation[block] = g
            balance[block] = b
            st.write_json(checkpoint, {"hours": block.stop})
            year = data.time[block.start].year
            if progress and year != last_year:
                print(f"  operations {year}: {block.stop:,}/{len(data.time):,} hours")
                last_year = year
        capacity = st.quantile_columns(
            balance, scenario.capacity_quantile, negative=True, memory_gib=memory_gib
        )
        annual = []
        totals = {
            name: np.zeros(len(nodes))
            for name in ("load", "wind", "solar", "backup", "curtailment", "export", "import")
        }
        for year in pd.unique(data.time.year):
            positions = np.flatnonzero(data.time.year == year)
            sums = {name: np.zeros(len(nodes)) for name in totals}
            for local in st.slices(len(positions)):
                block = slice(int(positions[local.start]), int(positions[local.stop - 1]) + 1)
                g, b = generation[block], balance[block]
                load = np.pad(data.load(block), ((0, 0), (0, len(nodes) - len(data.nodes))))
                injection = g.sum(axis=2) - load - b
                values = {
                    "load": load,
                    "wind": g[:, :, 0],
                    "solar": g[:, :, 1],
                    "backup": np.maximum(-b, 0),
                    "curtailment": np.maximum(b, 0),
                    "export": np.maximum(injection, 0),
                    "import": np.maximum(-injection, 0),
                }
                for name, v in values.items():
                    sums[name] += v.sum(axis=0)
            for name, v in sums.items():
                totals[name] += v
            annual.append(
                pd.DataFrame(
                    {
                        "year": int(year),
                        "node_id": nodes.node_id,
                        "country": nodes.country,
                        **{f"{k}_mwh": v for k, v in sums.items()},
                    }
                )
            )
        pd.concat(annual, ignore_index=True).to_parquet(path / "annual.parquet", index=False)
        st.save_arrays(path / "totals.npz", backup_capacity=capacity, **totals)
        st.write_json(path / "complete.json", definition)
    return path


def injection_block(data, nodes, root, block):
    generation = root["generation"][block]
    load = np.pad(data.load(block), ((0, 0), (0, len(nodes) - len(data.nodes))))
    balance = root["balancing"][block]
    return generation.sum(axis=2) - load - balance


def power_flow(data, scenario, layout_path, operation_path, *, memory_gib=24, progress=True):
    nodes = pd.read_parquet(layout_path / "nodes.parquet")
    links = build_links(nodes, data.links, scenario)
    definition = {
        "operation": operation_path.name,
        "links": links.to_dict(orient="list"),
        "quantile": scenario.capacity_quantile,
        "weight": scenario.length_exponent,
        "bounds": scenario.length_exponent_bounds,
        "algorithm": 3,
    }
    path = data.project / ".eurotracer" / "flow" / st.identity(definition)
    with st.lock(path):
        if (path / "complete.json").exists():
            if progress:
                print("Reusing network flow and capacities")
            return path
        op = st.group(operation_path / "arrays.zarr")
        n, link_count = len(nodes), len(links)
        lengths = links.length_km.to_numpy()
        exponent = scenario.length_exponent
        history = []
        if exponent == "minimize_volume":
            if progress:
                print("Selecting electrical length weighting from exact flow quantiles")
            chosen = path / "selected.json"
            if chosen.exists():
                selection = st.read_json(chosen)
                exponent, history = selection["exponent"], selection["history"]
            else:
                candidate_path = path / "candidate.npy"
                candidate = np.lib.format.open_memmap(
                    candidate_path, mode="w+", dtype="f8", shape=(len(data.time), link_count)
                )

                def objective(exponent):
                    weights = lengths ** (-exponent)
                    model = Network(n, links.source, links.destination, weights / weights.mean())
                    for block in st.slices(len(data.time)):
                        candidate[block] = model.solve(injection_block(data, nodes, op, block))[0]
                    capacities = st.quantile_columns(
                        candidate, scenario.capacity_quantile, absolute=True, memory_gib=memory_gib
                    )
                    volume = float(capacities @ lengths)
                    history.append({"exponent": float(exponent), "volume_mw_km": volume})
                    if progress:
                        print(f"  exponent={exponent:.4f}, volume={volume:,.0f} MW km")
                    return volume

                try:
                    low, high = scenario.length_exponent_bounds
                    objective(low)
                    objective(high)
                    fit = minimize_scalar(
                        objective, bounds=(low, high), method="bounded", options={"xatol": 1e-3}
                    )
                    if not fit.success:
                        raise ArithmeticError("Length weighting optimizer did not converge")
                    exponent = min(history, key=lambda row: row["volume_mw_km"])["exponent"]
                    st.write_json(chosen, {"exponent": exponent, "history": history})
                finally:
                    candidate._mmap.close()
                    candidate_path.unlink(missing_ok=True)
        exponent = float(exponent)
        weights = lengths ** (-exponent)
        if link_count:
            weights /= weights.mean()
        model = Network(n, links.source, links.destination, weights)
        root = st.group(path / "arrays.zarr", "a")
        flow = st.array(root, "flow", (len(data.time), link_count))
        checkpoint = path / "progress.json"
        state = (
            st.read_json(checkpoint)
            if checkpoint.exists()
            else {"hours": 0, "max_residual_mw": 0.0}
        )
        maximum = state["max_residual_mw"]
        if progress:
            print(
                f"Solving DC flow: {link_count} links, {len(model.parts)} components, "
                f"exponent={exponent:.4f}"
            )
        for block in st.slices(len(data.time)):
            if block.stop <= state["hours"]:
                continue
            f, _, error = model.solve(injection_block(data, nodes, op, block))
            flow[block] = f
            maximum = max(maximum, error)
            st.write_json(checkpoint, {"hours": block.stop, "max_residual_mw": maximum})
        links["susceptance"] = weights
        links["capacity_mw"] = (
            st.quantile_columns(
                flow, scenario.capacity_quantile, absolute=True, memory_gib=memory_gib
            )
            if link_count
            else []
        )
        links.to_parquet(path / "links.parquet", index=False)
        st.save_arrays(path / "network.npz", component=model.component)
        st.write_json(
            path / "complete.json",
            {
                "operation": operation_path.name,
                "exponent": exponent,
                "history": history,
                "max_residual_mw": maximum,
            },
        )
    return path
