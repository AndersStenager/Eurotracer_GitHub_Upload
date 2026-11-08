"""Input validation, snapshots, calendar alignment and explicit yearly CF reads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import storage as st
from .config import Weather


def timeline(years=None, start=None, stop=None):
    if years is not None:
        if start is not None or stop is not None or len(years) != 2:
            raise ValueError("Supply years=(first,last) or start/stop, not both")
        if any(not isinstance(y, (int, np.integer)) or isinstance(y, bool) for y in years):
            raise ValueError("Year bounds must be integers")
        start, stop = f"{int(years[0])}-01-01", f"{int(years[1]) + 1}-01-01"
    if start is None or stop is None:
        raise ValueError("Specify inclusive years=(first,last) or exclusive start/stop")
    a, b = [
        pd.Timestamp(x).tz_localize("UTC")
        if pd.Timestamp(x).tz is None
        else pd.Timestamp(x).tz_convert("UTC")
        for x in (start, stop)
    ]
    if b <= a or a != a.floor("h") or b != b.floor("h"):
        raise ValueError("Time bounds must be increasing, whole UTC hours")
    return pd.date_range(a.tz_localize(None), b.tz_localize(None), freq="h", inclusive="left")


def load_template(path, ids):
    table = pd.read_csv(path)
    timestamps = [c for c in table if c.lower() in ("time", "timestamp", "datetime")]
    if len(timestamps) > 1:
        raise ValueError("load_template.csv has multiple time columns")
    if timestamps:
        times = pd.DatetimeIndex(pd.to_datetime(table.pop(timestamps[0]), utc=True)).tz_localize(
            None
        )
    else:
        if set(table.columns) != set(ids):
            table = pd.read_csv(path, header=None)
            if table.shape[1] != len(ids):
                raise ValueError("Headerless load must have one column per bus")
            table.columns = ids
        times = pd.date_range(
            "2012-01-01" if len(table) == 8784 else "2013-01-01", periods=len(table), freq="h"
        )
    if set(table.columns) != set(ids) or len(table) not in (8760, 8784):
        raise ValueError("Load template must contain one complete year and match bus IDs")
    if (
        times.has_duplicates
        or not np.all(np.diff(times.to_numpy(dtype="datetime64[ns]")) == np.timedelta64(1, "h"))
        or times[0].month != 1
        or times[0].day != 1
        or times[0].hour != 0
        or times[-1].month != 12
        or times[-1].day != 31
        or times[-1].hour != 23
    ):
        raise ValueError("Load template must be an uninterrupted UTC calendar year")
    table = table.loc[:, list(ids)].astype(float)
    if not np.isfinite(table.to_numpy()).all() or (table.to_numpy() < 0).any():
        raise ValueError("Load must be finite and nonnegative")
    if table.to_numpy().sum() <= 0:
        raise ValueError("Load template must have positive total demand")
    table.index = times
    return table


def expand_load(template, time):
    source = template.index
    keys = pd.MultiIndex.from_arrays((source.month, source.day, source.hour))
    target = pd.MultiIndex.from_arrays((time.month, time.day, time.hour))
    positions = keys.get_indexer(target)
    result = template.to_numpy()[np.maximum(positions, 0)].copy()
    missing = positions < 0
    if missing.any():
        if not np.all((time[missing].month == 2) & (time[missing].day == 29)):
            raise ValueError("Missing non-leap calendar hours in load template")
        hours = time[missing].hour
        before = keys.get_indexer([(2, 28, h) for h in hours])
        after = keys.get_indexer([(3, 1, h) for h in hours])
        result[missing] = 0.5 * (template.to_numpy()[before] + template.to_numpy()[after])
    return result


def read_network(path):
    nodes = pd.read_csv(Path(path) / "bus.csv", dtype={"name": str, "country": str})
    nodes = nodes.rename(columns={"name": "node_id", "x": "longitude", "y": "latitude"})
    for c in ("node_id", "country"):
        if c not in nodes or nodes[c].isna().any() or nodes[c].str.strip().eq("").any():
            raise ValueError(f"bus.csv requires nonempty {c}")
        nodes[c] = nodes[c].str.strip()
    if nodes.empty or nodes.node_id.duplicated().any():
        raise ValueError("bus.csv node IDs must be unique and nonempty")
    if (
        not np.isfinite(nodes[["longitude", "latitude"]]).all().all()
        or not nodes.longitude.between(-180, 180).all()
        or not nodes.latitude.between(-90, 90).all()
    ):
        raise ValueError("Invalid bus coordinates")
    nodes = nodes[["node_id", "longitude", "latitude", "country"]].copy()
    nodes["is_offshore"] = False
    links = pd.read_csv(Path(path) / "links.csv", dtype=str)
    for c in ("link_id", "source_node_id", "destination_node_id"):
        if c not in links or links[c].isna().any() or links[c].str.strip().eq("").any():
            raise ValueError(f"links.csv requires nonempty {c}")
    if links.link_id.duplicated().any():
        raise ValueError("Duplicate link IDs")
    ids = pd.Index(nodes.node_id)
    s, d = (ids.get_indexer(links[c]) for c in ("source_node_id", "destination_node_id"))
    if (s < 0).any() or (d < 0).any() or (s == d).any():
        raise ValueError("Links must connect two different known nodes")
    if len(set(zip(np.minimum(s, d), np.maximum(s, d), strict=True))) != len(links):
        raise ValueError("Duplicate undirected links")
    return nodes, links


def validate_sites(sites, nodes):
    required = {
        "site_id",
        "node_id",
        "longitude",
        "latitude",
        "cell_area_km2",
        "land_wind_area_km2",
        "solar_area_km2",
        "is_sea",
    }
    if not required.issubset(sites) or sites.empty or sites.site_id.duplicated().any():
        raise ValueError(f"Sites require unique IDs and columns {sorted(required)}")
    sites = sites.copy()
    sites["node_id"] = sites.node_id.fillna("").astype(str)
    sites["site_id"] = sites.site_id.astype(str)
    if sites.site_id.str.strip().eq("").any() or not pd.api.types.is_bool_dtype(sites.is_sea):
        raise ValueError("Site IDs must be nonempty and is_sea must be boolean")
    if not np.isfinite(sites[["longitude", "latitude"]]).all().all():
        raise ValueError("Weather site coordinates must be finite")
    sites["node"] = pd.Index(nodes.node_id).get_indexer(sites.node_id)
    if ((sites.node < 0) & (sites.node_id != "")).any():
        raise ValueError("Weather sites reference unknown node IDs")
    for name, default in (("basin_id", ""), ("water_depth_m", 0.0), ("distance_to_land_m", 0.0)):
        if name not in sites:
            sites[name] = default
    sites["basin_id"] = sites.basin_id.fillna("").astype(str)
    for name in (
        "cell_area_km2",
        "land_wind_area_km2",
        "solar_area_km2",
        "water_depth_m",
        "distance_to_land_m",
    ):
        if not np.isfinite(sites[name]).all() or (sites[name] < 0).any():
            raise ValueError(f"Invalid site {name}")
    if (
        sites[["land_wind_area_km2", "solar_area_km2"]].to_numpy()
        > sites.cell_area_km2.to_numpy()[:, None] + 1e-7
    ).any():
        raise ValueError("Eligible area exceeds weather cell area")
    missing = set(range(len(nodes))) - set(sites.loc[sites.node >= 0, "node"])
    if missing:
        raise ValueError(f"Nodes have no mapped weather pixels: {sorted(missing)}")
    return sites


@dataclass
class Prepared:
    project: Path
    path: Path
    source: Path
    archive: Path
    time: pd.DatetimeIndex
    nodes: pd.DataFrame
    links: pd.DataFrame
    sites: pd.DataFrame
    template: pd.DataFrame
    meta: dict

    @property
    def id(self):
        return self.meta["id"]

    def load(self, selection):
        return expand_load(self.template, self.time[selection])

    def cf(self, selection):
        """Only the selected hours are read; no multi-year xarray concat."""
        if not isinstance(selection, slice) or selection.step not in (None, 1):
            raise ValueError("CF reads require a contiguous hourly slice")
        times = self.time[selection]
        wind = np.empty((len(times), len(self.sites)), dtype=self.meta["cf_dtype"])
        solar = np.empty_like(wind)
        for year in pd.unique(times.year):
            positions = np.flatnonzero(times.year == year)
            source = st.group(Path(self.meta["years_root"]) / str(year) / "arrays.zarr")
            offset = int(
                (times[positions[0]] - pd.Timestamp(f"{year}-01-01")) / pd.Timedelta(hours=1)
            )
            rows = slice(offset, offset + len(positions))
            wind[positions] = source["wind"][rows]
            solar[positions] = source["solar"][rows]
        return wind, solar


def open_prepared(path):
    path = Path(path).resolve()
    meta = st.read_json(path / "prepared.json")
    source = Path(meta["source"])
    nodes, links = read_network(source)
    archive = Path(meta["archive"])
    sites = validate_sites(pd.read_parquet(path / "sites.parquet"), nodes)
    return Prepared(
        Path(meta["project"]),
        path,
        source,
        archive,
        timeline(start=meta["start"], stop=meta["stop"]),
        nodes,
        links,
        sites,
        load_template(source / "load_template.csv", nodes.node_id),
        meta,
    )


def prepare(
    project,
    *,
    inputs="input1",
    years=None,
    start=None,
    stop=None,
    weather=None,
    archive=None,
    offshore=False,
    download=True,
    progress=True,
):
    """Prepare immutable input snapshots and required weather hours automatically.

    An explicit archive contains sites.parquet, archive.json, and yearly arrays.
    Use import_weather for independently prepared or legacy capacity factors.
    """
    project = Path(project).resolve()
    inputs = st.safe_name(inputs)
    source = project / "inputs" / inputs
    times = timeline(years, start, stop)
    weather = weather or Weather()
    for name in ("bus.csv", "links.csv", "load_template.csv"):
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    snap, hashes = st.snapshot(
        source,
        project / ".eurotracer" / "inputs",
        ("bus.csv", "links.csv", "load_template.csv", "offshore_basins.geojson", "costs_2030.csv"),
    )
    nodes, _ = read_network(snap)
    load_template(snap / "load_template.csv", nodes.node_id)
    if archive is None:
        from .weather import acquire

        archive = acquire(
            project,
            inputs,
            snap,
            nodes,
            times,
            weather,
            offshore=offshore,
            download=download,
            progress=progress,
        )
    archive = Path(archive).resolve()
    info = st.read_json(archive / "archive.json")
    if info.get("weather") != st.plain(weather):
        raise ValueError("Archive conversion assumptions differ from requested Weather")
    sites = validate_sites(pd.read_parquet(archive / "sites.parquet"), nodes)
    years_root = Path(info.get("years_root", archive / "years"))
    slices_info = []
    dtypes = []
    for year in pd.unique(times.year):
        yearpath = years_root / str(year)
        metadata = st.read_json(yearpath / "complete.json")
        requested = times[times.year == year]
        begin = int((requested[0] - pd.Timestamp(f"{year}-01-01")) / pd.Timedelta(hours=1))
        end = begin + len(requested)
        if begin < metadata.get("start", 0) or end > metadata["stop"]:
            raise ValueError(f"Weather archive lacks requested hours in {year}")
        selected_chunks = [
            c for c in metadata.get("chunks", []) if c["stop"] > begin and c["start"] < end
        ]
        if "chunks" in metadata:
            covered = begin
            for chunk in sorted(selected_chunks, key=lambda c: c["start"]):
                if chunk["start"] > covered:
                    break
                covered = max(covered, chunk["stop"])
            if covered < end:
                raise ValueError(f"Weather archive has a gap in {year}")
        z = st.group(yearpath / "arrays.zarr")
        dtypes.extend([z["wind"].dtype, z["solar"].dtype])
        expected = (len(timeline(years=(int(year), int(year)))), len(sites))
        if z["wind"].shape != expected or z["solar"].shape != expected:
            raise ValueError(f"Weather shape mismatch in {year}")
        if metadata["sites"] != info["sites"]:
            raise ValueError("Weather site coordinate identity differs from archive")
        slices_info.append(
            {
                "year": int(year),
                "id": st.identity(selected_chunks) if selected_chunks else metadata["id"],
                "start": begin,
                "stop": end,
            }
        )
    physical = {
        k: v
        for k, v in hashes.items()
        if k != "costs_2030.csv" and (offshore or k != "offshore_basins.geojson")
    }
    definition = {
        "inputs": physical,
        "sites": st.file_hash(archive / "sites.parquet"),
        "weather": slices_info,
        "start": str(times[0]),
        "stop": str(times[-1] + pd.Timedelta(hours=1)),
        "format": st.FORMAT,
        "cf_dtype": np.result_type(*dtypes).str,
    }
    key = st.identity(definition)
    # Snapshot pointer also distinguishes price-only revisions, without changing physics ID.
    path = project / ".eurotracer" / "prepared" / st.identity([key, str(snap)])
    with st.lock(path):
        if not (path / "prepared.json").exists():
            sites.to_parquet(path / "sites.parquet", index=False)
            st.write_json(
                path / "prepared.json",
                {
                    **definition,
                    "id": key,
                    "project": project,
                    "source": snap,
                    "archive": archive,
                    "years_root": years_root,
                },
            )
    if progress:
        print(f"Inputs ready: {len(nodes)} nodes, {len(sites)} pixels, {len(times):,} UTC hours")
    return open_prepared(path)


def import_weather(destination, sites, years, *, weather=None, provenance=None):
    """Import aligned annual CFs once, blockwise, without loading a whole archive.

    years maps year -> (wind, solar); arrays must be time-by-site with a complete
    UTC year. NumPy memmaps, Zarr arrays and other sliceable arrays are accepted.
    Site order is explicitly the row order of the supplied sites table.
    """
    destination = Path(destination).resolve()
    weather = weather or Weather()
    sites = pd.read_parquet(sites) if isinstance(sites, (str, Path)) else sites.copy()
    ids = sites.site_id.astype(str).tolist()
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError("Imported site IDs must be nonempty and unique")
    with st.lock(destination):
        site_id = st.identity(sites.fillna("").to_dict(orient="list"))
        info = {"sites": site_id, "weather": st.plain(weather), "provenance": provenance or {}}
        if (destination / "archive.json").exists():
            if st.read_json(destination / "archive.json") != info:
                raise ValueError("Use a new archive directory for changed imported inputs")
        else:
            sites.to_parquet(destination / "sites.parquet", index=False)
            st.write_json(destination / "archive.json", info)
        for year, (wind, solar) in years.items():
            shape = (len(timeline(years=(year, year))), len(sites))
            if tuple(wind.shape) != shape or tuple(solar.shape) != shape:
                raise ValueError(f"Imported {year} arrays must have shape {shape}")
            path = destination / "years" / str(year)
            path.mkdir(parents=True, exist_ok=True)
            complete = path / "complete.json"
            old = st.read_json(complete) if complete.exists() else None
            root = st.group(path / "arrays.zarr", "a")
            dtypes = [np.dtype(getattr(v, "dtype", np.float64)) for v in (wind, solar)]
            if any(d.kind != "f" or d.itemsize not in (4, 8) for d in dtypes):
                raise ValueError("CF imports require float32 or float64 arrays")
            out = [
                st.array(root, n, shape, dtype=d)
                for n, d in zip(("wind", "solar"), dtypes, strict=True)
            ]
            import hashlib

            digest = hashlib.sha256()
            for block in st.slices(shape[0]):
                for target, values in zip(out, (wind, solar), strict=True):
                    x = np.asarray(values[block], dtype=target.dtype)
                    if not np.isfinite(x).all() or (x < 0).any():
                        raise ValueError(f"Invalid capacity factors in {year}, hour {block.start}")
                    digest.update(x.tobytes())
                    if old is None:
                        target[block] = x
            if old is not None and old["id"] != digest.hexdigest():
                raise ValueError(f"Changed CF values for {year}; use a new archive directory")
            st.write_json(
                path / "complete.json",
                {"id": digest.hexdigest(), "sites": site_id, "start": 0, "stop": shape[0]},
            )
    return destination
