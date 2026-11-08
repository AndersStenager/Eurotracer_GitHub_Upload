"""Monthly ERA5 acquisition, yearly CF storage, and stable geographic facts."""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import geography
from . import storage as st
from .data import timeline

LAND_CODES = (
    111,
    112,
    121,
    122,
    123,
    124,
    131,
    132,
    133,
    141,
    142,
    211,
    212,
    213,
    221,
    222,
    223,
    231,
    241,
    242,
    243,
    244,
    311,
    312,
    313,
    321,
    322,
    323,
    324,
    331,
    332,
    333,
    334,
    335,
)


def validate_basins(path):
    import geopandas as gpd

    raw = st.read_json(path)
    if raw.get("features") == []:
        return gpd.GeoDataFrame({"basin_id": [], "name": []}, geometry=[], crs=4326)
    basins = gpd.read_file(path)
    if (
        not {"basin_id", "name", "geometry"}.issubset(basins)
        or basins.crs is None
        or basins.crs.to_epsg() != 4326
    ):
        raise ValueError("Basins need basin_id, name and EPSG:4326 polygon geometry")
    if (
        basins.basin_id.isna().any()
        or basins.basin_id.duplicated().any()
        or basins.basin_id.astype(str).str.strip().eq("").any()
        or (~basins.geometry.is_valid).any()
        or basins.geometry.is_empty.any()
        or (~basins.geom_type.isin(("Polygon", "MultiPolygon"))).any()
    ):
        raise ValueError("Invalid basin IDs or geometry")
    projected = basins.to_crs(3035)
    for i in range(len(basins)):
        for j in range(i):
            if projected.geometry.iloc[i].intersection(projected.geometry.iloc[j]).area > 1:
                raise ValueError("Basin interiors overlap")
    return basins


def static_sites(cutout, nodes, countries, corine, gebco, basins, weather):
    import atlite
    import geopandas as gpd
    import rasterio
    import shapely
    from atlite.gis import ExclusionContainer
    from shapely.geometry import box

    x = cutout.data.x.to_numpy()
    y = cutout.data.y.to_numpy()
    yy, xx = np.meshgrid(y, x, indexing="ij")
    lon, lat = xx.ravel(), yy.ravel()
    area = cutout.area(crs=3035).transpose("y", "x").to_numpy().ravel() / 1e6
    fractions = []
    for kind, excluded in (
        ("wind", weather.wind_excluded_codes),
        ("solar", weather.solar_excluded_codes),
    ):
        excluder = ExclusionContainer(crs=3035, res=weather.excluder_resolution_m)
        excluder.add_raster(str(corine), codes=list(LAND_CODES), invert=True)
        excluder.add_raster(str(corine), codes=list(excluded))
        if kind == "wind":
            excluder.add_raster(str(corine), codes=[111], buffer=weather.continuous_urban_setback_m)
            excluder.add_raster(
                str(corine), codes=[112], buffer=weather.discontinuous_urban_setback_m
            )
        fraction = np.zeros((len(y), len(x)))
        # Bound the high-resolution exclusion raster independently of European extent.
        for ys in st.slices(len(y), 32):
            for xs in st.slices(len(x), 32):
                tile = atlite.Cutout(
                    path=cutout.path.parent / ".tile.nc", data=cutout.data.isel(y=ys, x=xs)
                )
                domain = gpd.GeoSeries(
                    [
                        box(
                            float(x[xs].min()) - 0.125,
                            float(y[ys].min()) - 0.125,
                            float(x[xs].max()) + 0.125,
                            float(y[ys].max()) + 0.125,
                        )
                    ],
                    crs=4326,
                ).to_crs(3035)
                values = tile.availabilitymatrix(domain, excluder).isel(dim_0=0)
                fraction[ys, xs] = values.fillna(0).clip(0, 1).transpose("y", "x").to_numpy()
        fractions.append(fraction.ravel())
    world = gpd.read_file(countries)
    # Nearby land suffices for masking and offshore setbacks; avoid worldwide distances.
    rectangle = box(float(x.min()) - 5, float(y.min()) - 5, float(x.max()) + 5, float(y.max()) + 5)
    world = world.to_crs(4326).clip(rectangle).to_crs(3035).geometry.union_all()
    points = gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs=4326).to_crs(3035)
    shapely.prepare(world)
    land = np.asarray(shapely.intersects(world, points.to_numpy()))
    depth = np.zeros(len(lon))
    if gebco is not None:
        with rasterio.open(gebco) as raster:
            depth = np.maximum(
                -np.array([v[0] for v in raster.sample(zip(lon, lat, strict=True))], dtype=float), 0
            )
    buses = nodes.rename(columns={"node_id": "name", "longitude": "x", "latitude": "y"})
    mapping = geography.build_site_voronoi_mapping(
        buses, lon, lat, countries_path=countries, onshore_mask=land
    )
    sites = pd.DataFrame(
        {
            "site_id": [f"era5-{a:.5f}-{b:.5f}" for a, b in zip(lon, lat, strict=True)],
            "node_id": mapping.voronoi_node_id,
            "longitude": lon,
            "latitude": lat,
            "cell_area_km2": area,
            "land_wind_area_km2": area * fractions[0] * land,
            "solar_area_km2": area * fractions[1] * land,
            "is_sea": ~land,
            "water_depth_m": depth,
            "distance_to_land_m": (
                points.distance(world).to_numpy() if gebco is not None else np.zeros(len(points))
            ),
        }
    )
    if basins is not None:
        basinmap = geography.build_site_basin_mapping(basins, lon, lat)
        sites["basin_id"] = basinmap.basin_id
        sites["basin_overlap_km2"] = basinmap.basin_overlap_km2
    return sites


def acquire(project, inputs, source, nodes, times, weather, *, offshore, download, progress):
    try:
        import atlite
    except ImportError as exc:
        raise RuntimeError(
            "Weather preparation needs pip install eurotracer[data], or import_weather"
        ) from exc
    from .downloads import sources

    countries, corine, gebco, source_ids = sources(
        project, inputs, download=download, offshore=False
    )
    buses = nodes.rename(columns={"node_id": "name", "longitude": "x", "latitude": "y"})
    cells = geography.build_voronoi_cells(buses, countries_path=countries).to_crs(4326)
    bounds = cells.total_bounds
    basinpath = source / "offshore_basins.geojson" if offshore else None
    if basinpath is not None:
        if not basinpath.is_file():
            raise FileNotFoundError("Offshore scenarios require offshore_basins.geojson")
        basins = validate_basins(basinpath)
        if not basins.empty:
            bb = basins.total_bounds
            bounds = np.r_[np.minimum(bounds[:2], bb[:2]), np.maximum(bounds[2:], bb[2:])]
    bounds = (np.floor(bounds / 0.25) * 0.25) + np.array(
        [-1, -1, 1, 1]
    ) * weather.domain_buffer_degrees
    # Round upper bounds outward, independently of model policy.
    bounds[2:] += 0.25
    bounds = tuple(float(v) for v in bounds)
    if offshore:
        _, _, gebco, source_ids = sources(
            project, inputs, download=download, offshore=True, bounds=bounds
        )
    domain_id = st.identity(
        {"bounds": bounds, "atlite": atlite.__version__, "grid": "ERA5 .25 degree"}
    )
    root = project / ".eurotracer" / "weather"
    raw_root = root / "era5" / domain_id
    raw_root.mkdir(parents=True, exist_ok=True)
    conversion = {
        k: asdict(weather)[k]
        for k in ("turbine", "hub_height_m", "smooth", "interpolation", "panel")
    }
    cf_root = (
        root
        / "cf"
        / st.identity({"domain": domain_id, "conversion": conversion, "atlite": atlite.__version__})
    )
    geographic_definition = {
        "domain": domain_id,
        "sources": source_ids,
        "weather": asdict(weather),
        "bus": st.file_hash(source / "bus.csv"),
        "basins": st.file_hash(basinpath) if basinpath is not None else None,
    }
    # Geometry revisions must invalidate site ownership and downstream model
    # caches, but not the coordinate-aligned ERA5/CF cache.
    previous_archive = root / "archives" / st.identity(geographic_definition)
    geo_id = st.identity({**geographic_definition, "site_mapping_algorithm": 2})
    archive = root / "archives" / geo_id
    with st.lock(archive):
        if not (archive / "archive.json").exists():
            if (previous_archive / "archive.json").exists():
                # Only ownership changed. Reuse the already screened areas and
                # exact site axis instead of repeating raster screening/downloads.
                if progress:
                    print("Updating coastal node ownership using cached screened pixels")
                sites = pd.read_parquet(previous_archive / "sites.parquet")
                mapping = geography.build_site_voronoi_mapping(
                    buses,
                    sites.longitude.to_numpy(),
                    sites.latitude.to_numpy(),
                    countries_path=countries,
                    onshore_mask=~sites.is_sea.to_numpy(dtype=bool),
                )
                sites["node_id"] = mapping.voronoi_node_id.to_numpy()
            else:
                if progress:
                    print("Preparing country ownership and screened pixel areas")
                with tempfile.TemporaryDirectory(dir=raw_root, prefix="grid-") as temporary:
                    cutout = atlite.Cutout(
                        path=Path(temporary) / "grid.nc",
                        module="era5",
                        bounds=bounds,
                        time=slice(times[0], times[0]),
                    )
                    sites = static_sites(
                        cutout, nodes, countries, corine, gebco, basinpath, weather
                    )
            sites.to_parquet(archive / "sites.parquet", index=False)
            axis_id = st.identity(
                sites[["site_id", "longitude", "latitude"]].to_dict(orient="list")
            )
            st.write_json(
                archive / "archive.json",
                {
                    "sites": axis_id,
                    "weather": weather,
                    "years_root": str(cf_root / "years"),
                    "bounds": bounds,
                    "sources": source_ids,
                    "era5_root": str(raw_root),
                    "provenance": {"atlite": atlite.__version__, "site_mapping_algorithm": 2},
                },
            )
    info = st.read_json(archive / "archive.json")
    sites = pd.read_parquet(archive / "sites.parquet")
    with st.lock(cf_root):
        for year in pd.unique(times.year):
            year = int(year)
            path = cf_root / "years" / str(year)
            path.mkdir(parents=True, exist_ok=True)
            meta = (
                st.read_json(path / "complete.json")
                if (path / "complete.json").exists()
                else {"sites": info["sites"], "chunks": []}
            )
            if meta["sites"] != info["sites"]:
                raise ValueError("Weather coordinate identity mismatch")
            months = sorted(set(times[times.year == year].month))
            for month in months:
                a = pd.Timestamp(year=year, month=int(month), day=1)
                b = a + pd.offsets.MonthBegin(1)
                begin = int((a - pd.Timestamp(f"{year}-01-01")) / pd.Timedelta(hours=1))
                stop = begin + int((b - a) / pd.Timedelta(hours=1))
                if any(c["start"] == begin and c["stop"] == stop for c in meta["chunks"]):
                    continue
                raw = raw_root / f"{year}-{month:02}.nc"
                if not download and not raw.is_file():
                    raise FileNotFoundError(
                        f"Missing ERA5 {year}-{month:02}; enable download or supply an archive"
                    )
                if progress:
                    print(
                        f"ERA5/CF {year}-{month:02}: "
                        f"{'reuse weather' if raw.is_file() else 'download'}"
                    )
                cutout = (
                    atlite.Cutout(path=raw)
                    if raw.is_file()
                    else atlite.Cutout(
                        path=raw,
                        module="era5",
                        bounds=bounds,
                        time=slice(a, b - pd.Timedelta(hours=1)),
                    )
                )
                try:
                    if not set(("wind", "influx", "temperature")).issubset(
                        cutout.prepared_features.index.unique("feature")
                    ):
                        if not download:
                            raise FileNotFoundError(f"Incomplete ERA5 fields in {raw}")
                        cutout.prepare(
                            features=["wind", "influx", "temperature"],
                            monthly_requests=True,
                            concurrent_requests=False,
                            data_format="netcdf",
                            dask_kwargs={"scheduler": "synchronous"},
                        )
                    expected = timeline(start=a, stop=b)
                    if not np.array_equal(cutout.data.time.to_numpy(), expected.to_numpy()):
                        raise ValueError("ERA5 monthly timestamps differ from requested UTC hours")
                    z = st.group(path / "arrays.zarr", "a")
                    shape = (len(timeline(years=(year, year))), len(sites))
                    turbine = dict(atlite.resource.get_windturbineconfig(weather.turbine))
                    turbine["hub_height"] = weather.hub_height_m
                    panel = atlite.resource.get_solarpanelconfig(weather.panel)
                    digest = hashlib.sha256()
                    for block in st.slices(len(expected), 168):
                        view = atlite.Cutout(
                            path=raw_root / ".conversion-view.nc", data=cutout.data.isel(time=block)
                        )
                        converted = (
                            view.wind(
                                turbine=turbine,
                                smooth=weather.smooth,
                                interpolation_method=weather.interpolation,
                                aggregate_time=None,
                            ),
                            view.pv(
                                panel=panel,
                                orientation="latitude_optimal",
                                tracking=None,
                                clearsky_model=None,
                                aggregate_time=None,
                            ),
                        )
                        for name, values in zip(("wind", "solar"), converted, strict=True):
                            if set(values.dims) != {"time", "y", "x"}:
                                raise ValueError("Atlite conversion returned unexpected dimensions")
                            values = values.transpose("time", "y", "x")
                            yy, xx = np.meshgrid(
                                values.y.to_numpy(), values.x.to_numpy(), indexing="ij"
                            )
                            if not np.array_equal(
                                xx.ravel(), sites.longitude
                            ) or not np.array_equal(yy.ravel(), sites.latitude):
                                raise ValueError("Atlite conversion grid order changed")
                            x = (
                                values.compute(scheduler="synchronous")
                                .to_numpy()
                                .reshape(len(values.time), -1)
                            )
                            if not np.isfinite(x).all() or (x < 0).any():
                                raise ValueError("Capacity factors must be finite and nonnegative")
                            output = st.array(
                                z, name, shape, dtype=x.dtype, chunks=(24, min(len(sites), 1024))
                            )
                            output[begin + block.start : begin + block.stop] = x
                            digest.update(x.tobytes())
                    meta["chunks"].append({"start": begin, "stop": stop, "id": digest.hexdigest()})
                    meta["chunks"].sort(key=lambda c: c["start"])
                    meta["id"] = st.identity(meta["chunks"])
                    meta["start"] = min(c["start"] for c in meta["chunks"])
                    meta["stop"] = max(c["stop"] for c in meta["chunks"])
                    st.write_json(path / "complete.json", meta)
                finally:
                    cutout.data.close()
    return archive
