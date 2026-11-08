"""Preparation-time construction of the site-aligned Voronoi mapping."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

EUROPE_EQUAL_AREA_CRS = 3035
SQUARE_METRES_PER_KM2 = 1_000_000.0
EXCLUDED_NATURAL_EARTH_COUNTRIES = frozenset(
    {
        "Russia",
        "Moldova",
        "Ukraine",
        "Turkey",
        "Belarus",
        "Faeroe Is.",
        "Åland",
        "Jersey",
        "Guernsey",
        "Isle of Man",
        "Gibraltar",
        "Andorra",
        "Malta",
        "Monaco",
        "San Marino",
        "Vatican",
        "Iceland",
    }
)


def build_site_voronoi_mapping(
    buses: pd.DataFrame,
    site_longitude: NDArray[np.floating],
    site_latitude: NDArray[np.floating],
    *,
    countries_path: str | Path,
    onshore_mask: NDArray[np.bool_] | None = None,
) -> pd.DataFrame:
    """Return V1-compatible, country-clipped node ownership for each grid site."""

    import geopandas as gpd
    from shapely.geometry import box

    longitude = np.asarray(site_longitude, dtype=np.float64).reshape(-1)
    latitude = np.asarray(site_latitude, dtype=np.float64).reshape(-1)
    if longitude.shape != latitude.shape or longitude.size == 0:
        raise ValueError("Voronoi site coordinates must be non-empty and aligned")
    if not np.isfinite(longitude).all() or not np.isfinite(latitude).all():
        raise ValueError("Voronoi site coordinates must be finite")
    cells, nodes = _build_voronoi_cells_and_nodes(buses, countries_path=countries_path)

    x_values = pd.unique(longitude)
    y_values = pd.unique(latitude)
    if x_values.size < 2 or y_values.size < 2:
        raise ValueError("Voronoi mapping requires at least a two-by-two regular grid")
    dx = float(np.median(np.diff(np.sort(x_values))))
    dy = float(np.median(np.diff(np.sort(y_values))))
    pixel_geometry = [
        box(lon - 0.5 * dx, lat - 0.5 * dy, lon + 0.5 * dx, lat + 0.5 * dy)
        for lon, lat in zip(longitude, latitude, strict=True)
    ]
    pixels = gpd.GeoDataFrame(
        {"site_position": np.arange(longitude.size, dtype=np.int64)},
        geometry=pixel_geometry,
        crs=4326,
    ).to_crs(EUROPE_EQUAL_AREA_CRS)
    cells = cells.reset_index(drop=True)
    cells["cell_position"] = np.arange(len(cells), dtype=np.int64)
    candidates = gpd.sjoin(
        pixels[["site_position", "geometry"]],
        cells[
            [
                "cell_position",
                "node_position",
                "node_id",
                "country_id",
                "geometry",
            ]
        ],
        how="inner",
        predicate="intersects",
    ).drop(columns="index_right")
    candidates = candidates.merge(
        cells[["cell_position", "geometry"]].rename(columns={"geometry": "cell_geometry"}),
        on="cell_position",
        how="left",
    )
    candidates["overlap_area_m2"] = candidates.geometry.intersection(
        gpd.GeoSeries(candidates["cell_geometry"], crs=EUROPE_EQUAL_AREA_CRS)
    ).area.to_numpy(dtype=np.float64)
    candidates = candidates.loc[candidates["overlap_area_m2"] > 0.0].copy()
    best = (
        candidates.sort_values(
            ["site_position", "overlap_area_m2", "node_position"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .drop_duplicates("site_position", keep="first")
        .reset_index(drop=True)
    )
    if best.empty:
        raise RuntimeError("no weather site overlaps a Voronoi cell")

    best = _rescue_unassigned_nodes(best, pixels, nodes, onshore_mask=onshore_mask)
    node_id = np.full(longitude.size, "", dtype=object)
    country_id = np.full(longitude.size, "", dtype=object)
    overlap = np.zeros(longitude.size, dtype=np.float64)
    position = best["site_position"].to_numpy(dtype=np.int64)
    node_id[position] = best["node_id"].astype(str).to_numpy()
    country_id[position] = best["country_id"].astype(str).to_numpy()
    overlap[position] = best["overlap_area_m2"].to_numpy(dtype=np.float64) / 1_000_000.0
    return pd.DataFrame(
        {
            "voronoi_node_id": node_id,
            "voronoi_country": country_id,
            "voronoi_overlap_km2": overlap,
            "voronoi_mapped": node_id != "",
        }
    )


def build_site_basin_mapping(
    basins_path: str | Path,
    site_longitude: NDArray[np.floating],
    site_latitude: NDArray[np.floating],
) -> pd.DataFrame:
    """Assign each regular-grid site to the basin with its largest overlap."""

    import geopandas as gpd
    from shapely.geometry import box

    longitude = np.asarray(site_longitude, dtype=np.float64).reshape(-1)
    latitude = np.asarray(site_latitude, dtype=np.float64).reshape(-1)
    if longitude.shape != latitude.shape or longitude.size == 0:
        raise ValueError("basin site coordinates must be non-empty and aligned")
    raw = json.loads(Path(basins_path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and raw.get("features") == []:
        return pd.DataFrame(
            {
                "basin_id": np.full(longitude.size, "", dtype=object),
                "basin_name": np.full(longitude.size, "", dtype=object),
                "basin_overlap_km2": np.zeros(longitude.size),
                "basin_mapped": np.zeros(longitude.size, dtype=bool),
            }
        )
    basins = gpd.read_file(Path(basins_path))
    required = {"basin_id", "name", "geometry"}
    if not required.issubset(basins.columns) or basins.crs is None:
        raise ValueError("offshore basins require basin_id, name, geometry, and CRS")
    x_values = np.sort(pd.unique(longitude))
    y_values = np.sort(pd.unique(latitude))
    if x_values.size < 2 or y_values.size < 2:
        raise ValueError("basin mapping requires at least a two-by-two regular grid")
    dx = float(np.median(np.diff(x_values)))
    dy = float(np.median(np.diff(y_values)))
    pixels = gpd.GeoDataFrame(
        {"site_position": np.arange(longitude.size, dtype=np.int64)},
        geometry=[
            box(lon - 0.5 * dx, lat - 0.5 * dy, lon + 0.5 * dx, lat + 0.5 * dy)
            for lon, lat in zip(longitude, latitude, strict=True)
        ],
        crs=4326,
    ).to_crs(EUROPE_EQUAL_AREA_CRS)
    basins = basins.loc[:, ["basin_id", "name", "geometry"]].to_crs(EUROPE_EQUAL_AREA_CRS)
    basins = basins.rename(columns={"name": "basin_name"}).reset_index(drop=True)
    basins["basin_position"] = np.arange(len(basins), dtype=np.int64)
    candidates = gpd.sjoin(
        pixels[["site_position", "geometry"]],
        basins[["basin_position", "basin_id", "basin_name", "geometry"]],
        how="inner",
        predicate="intersects",
    ).drop(columns="index_right")
    candidates = candidates.merge(
        basins[["basin_position", "geometry"]].rename(columns={"geometry": "basin_geometry"}),
        on="basin_position",
        how="left",
    )
    candidates["overlap_area_m2"] = candidates.geometry.intersection(
        gpd.GeoSeries(candidates["basin_geometry"], crs=EUROPE_EQUAL_AREA_CRS)
    ).area.to_numpy(dtype=np.float64)
    best = (
        candidates.loc[candidates["overlap_area_m2"] > 0]
        .sort_values(
            ["site_position", "overlap_area_m2", "basin_id"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .drop_duplicates("site_position", keep="first")
    )
    basin_id = np.full(longitude.size, "", dtype=object)
    basin_name = np.full(longitude.size, "", dtype=object)
    overlap = np.zeros(longitude.size, dtype=np.float64)
    if not best.empty:
        position = best["site_position"].to_numpy(dtype=np.int64)
        basin_id[position] = best["basin_id"].astype(str).to_numpy()
        basin_name[position] = best["basin_name"].astype(str).to_numpy()
        overlap[position] = best["overlap_area_m2"].to_numpy() / SQUARE_METRES_PER_KM2
    return pd.DataFrame(
        {
            "basin_id": basin_id,
            "basin_name": basin_name,
            "basin_overlap_km2": overlap,
            "basin_mapped": basin_id != "",
        }
    )


def build_voronoi_cells(
    buses: pd.DataFrame,
    *,
    countries_path: str | Path,
) -> Any:
    """Return the exact country-clipped cells used by the site mapping."""

    cells, _ = _build_voronoi_cells_and_nodes(buses, countries_path=countries_path)
    return cells


def _build_voronoi_cells_and_nodes(
    buses: pd.DataFrame,
    *,
    countries_path: str | Path,
) -> tuple[Any, Any]:
    """Build country-clipped Voronoi cells and their owning node IDs."""
    import geopandas as gpd
    from shapely import make_valid, union_all, voronoi_polygons
    from shapely.geometry import MultiPoint
    from shapely.ops import nearest_points

    required = {"name", "x", "y", "country"}
    missing = required - set(buses.columns)
    if missing:
        raise ValueError(f"Voronoi buses are missing: {', '.join(sorted(missing))}")
    if buses["name"].astype(str).duplicated().any():
        raise ValueError("Voronoi bus names must be unique")
    if buses.duplicated(["x", "y"]).any():
        raise ValueError("Voronoi bus coordinates must be unique")

    world = gpd.read_file(Path(countries_path))
    boundary_columns = {"CONTINENT", "NAME", "geometry"}
    if not boundary_columns.issubset(world.columns) or world.crs is None:
        raise ValueError("Natural Earth countries lack CONTINENT, NAME, geometry, or CRS")
    if "ISO_A2_EH" in world.columns:
        requested = set(buses["country"].astype(str))
        available = set(world["ISO_A2_EH"].astype(str))
        missing_country = sorted(requested - available)
        if missing_country:
            raise ValueError(
                "bus country IDs are missing from Natural Earth ISO_A2_EH: "
                + ", ".join(missing_country)
            )
        countries = world.loc[
            world["ISO_A2_EH"].astype(str).isin(requested),
            ["NAME", "geometry"],
        ].to_crs(EUROPE_EQUAL_AREA_CRS)
    else:
        # Compatibility for small synthetic boundary fixtures created before
        # input-set country selection became part of the public contract.
        countries = world.loc[
            (world["CONTINENT"] == "Europe")
            & ~world["NAME"].isin(EXCLUDED_NATURAL_EARTH_COUNTRIES),
            ["NAME", "geometry"],
        ].to_crs(EUROPE_EQUAL_AREA_CRS)
    countries["geometry"] = countries.geometry.map(make_valid)
    countries = countries.explode(index_parts=False, ignore_index=True)
    countries = countries.loc[countries.geometry.notna() & ~countries.geometry.is_empty].copy()
    if countries.empty:
        raise ValueError("Natural Earth Europe boundary is empty")

    nodes = gpd.GeoDataFrame(
        {
            "node_position": np.arange(len(buses), dtype=np.int64),
            "node_id": buses["name"].astype(str).to_numpy(),
            "country_id": buses["country"].astype(str).to_numpy(),
        },
        geometry=gpd.points_from_xy(buses["x"], buses["y"]),
        crs=4326,
    ).to_crs(EUROPE_EQUAL_AREA_CRS)
    europe = make_valid(union_all(countries.geometry.to_numpy()))
    inside = nodes.geometry.within(europe)
    if not inside.all():
        inner = europe.buffer(-10.0)
        if inner.is_empty:
            inner = europe
        nodes.loc[~inside, "geometry"] = nodes.loc[~inside, "geometry"].map(
            lambda point: nearest_points(point, inner)[1]
        )

    rows: list[dict[str, Any]] = []
    for component in countries.geometry:
        members = nodes.loc[nodes.geometry.map(component.covers)].copy()
        if members.empty:
            continue
        if len(members) == 1:
            polygons = [component]
        else:
            diagram = voronoi_polygons(
                MultiPoint(members.geometry.to_list()),
                extend_to=component.envelope,
                ordered=True,
            )
            polygons = [make_valid(polygon.intersection(component)) for polygon in diagram.geoms]
        if len(polygons) != len(members):
            raise RuntimeError("Voronoi polygon count does not match its component nodes")
        for node, polygon in zip(members.itertuples(index=False), polygons, strict=True):
            if polygon is None or polygon.is_empty:
                continue
            rows.append(
                {
                    "node_position": int(node.node_position),
                    "node_id": str(node.node_id),
                    "country_id": str(node.country_id),
                    "geometry": polygon,
                }
            )
    cells = gpd.GeoDataFrame(rows, geometry="geometry", crs=EUROPE_EQUAL_AREA_CRS)
    if cells.empty:
        raise RuntimeError("Voronoi construction produced no cells")
    represented = set(cells["node_position"].astype(int))
    missing_nodes = sorted(set(range(len(nodes))) - represented)
    if missing_nodes:
        raise RuntimeError(f"Voronoi construction omitted node positions {missing_nodes}")
    return cells, nodes


def _rescue_unassigned_nodes(
    best: pd.DataFrame, pixels: Any, nodes: Any, *, onshore_mask=None
) -> pd.DataFrame:
    """Give every node a usable pixel without removing a donor's last one.

    When the land mask is supplied, sea-centred pixels cannot satisfy onshore
    coverage. Coastal nodes with only those pixels also need a land donor.
    Ownership changes; weather coordinates, screened areas and CFs do not.
    """

    result = best.copy()
    usable = np.ones(len(pixels), dtype=bool)
    if onshore_mask is not None:
        usable = np.asarray(onshore_mask, dtype=bool)
        if usable.shape != (len(pixels),):
            raise ValueError("Onshore mask must align with weather site positions")
    usable_rows = usable[result["site_position"].to_numpy(dtype=int)]
    present = set(result.loc[usable_rows, "node_position"].astype(int))
    missing = sorted(set(nodes["node_position"].astype(int)) - present)
    if not missing:
        return result
    centroids = pixels.geometry.centroid
    pixel_xy = pd.DataFrame(
        {
            "site_position": pixels["site_position"].to_numpy(dtype=np.int64),
            "x": centroids.x.to_numpy(dtype=np.float64),
            "y": centroids.y.to_numpy(dtype=np.float64),
        }
    )
    for node_position in missing:
        node = nodes.loc[nodes["node_position"] == node_position].iloc[0]
        usable_rows = usable[result["site_position"].to_numpy(dtype=int)]
        counts = result.loc[usable_rows, "node_position"].value_counts()
        donors = result.loc[usable_rows & (result["node_position"].map(counts) > 1)].copy()
        same_country = donors.loc[donors["country_id"].astype(str) == str(node.country_id)]
        if not same_country.empty:
            donors = same_country
        if donors.empty:
            raise RuntimeError("Voronoi node rescue has no donor pixel with spare usable coverage")
        donors = donors.merge(pixel_xy, on="site_position", how="left")
        donors["distance2"] = (donors["x"] - float(node.geometry.x)) ** 2 + (
            donors["y"] - float(node.geometry.y)
        ) ** 2
        donor_index = int(donors.sort_values(["distance2", "site_position"]).index[0])
        site_position = int(donors.loc[donor_index, "site_position"])
        target_rows = result.index[result["site_position"] == site_position]
        if target_rows.size != 1:
            raise RuntimeError("Voronoi node rescue donor is not unique")
        row = int(target_rows[0])
        result.loc[row, "node_position"] = int(node_position)
        result.loc[row, "node_id"] = str(node.node_id)
        result.loc[row, "country_id"] = str(node.country_id)
    return result
