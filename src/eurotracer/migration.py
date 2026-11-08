"""One-time data conversion. No old runtime code is imported or executed."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from . import storage as st
from .config import Weather
from .data import import_weather, timeline


def import_legacy_weather(
    view: str | Path, destination: str | Path, *, years: tuple[int, int], weather: Weather
) -> Path:
    """Validate and copy a V2 yearly-weather-v3 view into the new annual archive.

    Weather is required explicitly: old screening/conversion settings must not
    silently become the thesis baseline. Original source files remain untouched.
    Partial years and unknown formats are rejected. For independently prepared
    arrays use import_weather with explicit provenance instead.
    """
    import atlite

    view = Path(view).resolve()
    meta = st.read_json(view / "meta.json")
    if meta.get("format") != "yearly-weather-v3":
        raise ValueError("Legacy importer supports yearly-weather-v3 views only")
    configuration = meta["configuration"]
    checks = {
        "wind_hub_height_m": weather.hub_height_m,
        "wind_smooth": weather.smooth,
        "wind_interpolation_method": weather.interpolation,
        "wind_excluded_codes": list(weather.wind_excluded_codes),
        "solar_excluded_codes": list(weather.solar_excluded_codes),
        "urban_continuous_setback_m": weather.continuous_urban_setback_m,
        "urban_discontinuous_setback_m": weather.discontinuous_urban_setback_m,
        "excluder_resolution_m": weather.excluder_resolution_m,
    }

    def resource_id(resource):
        def default(value):
            return value.tolist() if isinstance(value, np.ndarray) else st.plain(value)

        encoded = json.dumps(resource, default=default, sort_keys=True, separators=(",", ":"))
        return (
            base64.urlsafe_b64encode(hashlib.sha256(encoded.encode()).digest()[:16])
            .decode()
            .rstrip("=")
        )

    turbine = dict(atlite.resource.get_windturbineconfig(weather.turbine))
    turbine["hub_height"] = weather.hub_height_m
    checks["wind_model_id"] = resource_id(turbine)
    checks["solar_model_id"] = resource_id(atlite.resource.get_solarpanelconfig(weather.panel))
    differences = [name for name, value in checks.items() if configuration.get(name) != value]
    if differences:
        raise ValueError(f"Legacy weather assumptions differ: {', '.join(differences)}")
    root = view.parents[2]
    sites = pd.read_parquet(root / meta["sites"] / "sites.parquet")
    mapping = pd.read_parquet(root / meta["mapping"] / "mapping.parquet")
    if sites.site_id.duplicated().any() or set(sites.site_id) != set(mapping.site_id):
        raise ValueError("Legacy site/mapping IDs differ or are duplicated")
    sites = sites.merge(mapping, on="site_id", how="left", sort=False, validate="one_to_one")
    sites["node_id"] = sites.voronoi_node_id.where(sites.voronoi_mapped, "")
    provenance = {
        "legacy_view": str(view),
        "view_sha256": st.file_hash(view / "meta.json"),
        "site_sha256": st.file_hash(root / meta["sites"] / "sites.parquet"),
        "mapping_sha256": st.file_hash(root / meta["mapping"] / "mapping.parquet"),
        "configuration": configuration,
    }
    for year in range(years[0], years[1] + 1):
        arrays = []
        expected = timeline(years=(year, year))
        origin, end = expected[0], expected[-1] + pd.Timedelta(hours=1)
        for technology in ("wind", "solar"):
            path = root / meta[technology] / "years" / str(year)
            info = st.read_json(path / "meta.json")
            if info.get("format") != "yearly-weather-v3" or info.get("year") != year:
                raise ValueError(f"Unknown legacy year metadata: {path}")
            cursor = origin
            for a, b in sorted((pd.Timestamp(a), pd.Timestamp(b)) for a, b in info["completed"]):
                if a > cursor:
                    break
                cursor = max(cursor, b)
            if cursor < end:
                raise ValueError(f"Incomplete legacy {technology} calendar year {year}")
            array = zarr.open_group(path / "arrays.zarr", mode="r")["capacity_factor"]
            if array.shape != (len(expected), len(sites)):
                raise ValueError(f"Legacy array axes differ in {path}")
            # Verify the old committed hashes with bounded reads before importing.
            covered = np.zeros(len(expected), dtype=bool)
            for interval, checksum in info["chunks"].items():
                a, b = (pd.Timestamp(x) for x in interval.split("/"))
                first = int((a - origin) / pd.Timedelta(hours=1))
                stop = int((b - origin) / pd.Timedelta(hours=1))
                if first < 0 or stop > len(expected) or stop <= first:
                    raise ValueError("Invalid legacy chunk interval")
                digest = hashlib.sha256()
                digest.update(b"capacity_factor")
                digest.update(str(array.dtype).encode())
                digest.update(np.asarray((stop - first, len(sites)), dtype=np.int64).tobytes())
                for block in st.slices(stop - first):
                    values = np.ascontiguousarray(array[first + block.start : first + block.stop])
                    digest.update(values.tobytes())
                if digest.hexdigest() != checksum:
                    raise ValueError(f"Legacy checksum mismatch in {path}: {interval}")
                covered[first:stop] = True
            if not covered.all():
                raise ValueError(f"Legacy checksums do not cover {year}")
            arrays.append(array)
        import_weather(
            destination, sites, {year: tuple(arrays)}, weather=weather, provenance=provenance
        )
    return Path(destination).resolve()
