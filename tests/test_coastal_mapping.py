import numpy as np
import pandas as pd
import pytest

from eurotracer.geography import _rescue_unassigned_nodes

gpd = pytest.importorskip("geopandas")
geometry = pytest.importorskip("shapely.geometry")
Point, box = geometry.Point, geometry.box


def test_coastal_rescue_preserves_each_donors_last_land_pixel():
    # Node 0 owns sea only. Node 1 has a nearby land pixel plus sea, so its
    # land pixel cannot be donated. Node 2 has two land pixels and may donate.
    pixels = gpd.GeoDataFrame(
        {"site_position": range(6)},
        geometry=[box(x - .1, -.1, x + .1, .1) for x in [0, 1, 2, 3, 4, 5]],
        crs=3035,
    )
    nodes = gpd.GeoDataFrame(
        {"node_position": [0, 1, 2], "node_id": ["coast", "single", "donor"],
         "country_id": ["AA"] * 3},
        geometry=[Point(0, 0), Point(1, 0), Point(4, 0)], crs=3035,
    )
    best = pd.DataFrame({
        "site_position": range(6), "node_position": [0, 1, 1, 2, 2, 2],
        "node_id": ["coast", "single", "single", "donor", "donor", "donor"],
        "country_id": ["AA"] * 6,
    })
    land = np.array([False, True, False, True, True, False])
    result = _rescue_unassigned_nodes(best, pixels, nodes, onshore_mask=land)
    assert result.loc[1, "node_id"] == "single"
    assert result.loc[3, "node_id"] == "coast"
    assert result.loc[4, "node_id"] == "donor"
    pd.testing.assert_series_equal(result.site_position, best.site_position)
    assert result.loc[land, "node_position"].value_counts().to_dict() == {1: 1, 0: 1, 2: 1}
    pd.testing.assert_frame_equal(
        result, _rescue_unassigned_nodes(result, pixels, nodes, onshore_mask=land)
    )
    # Legacy callers without a land mask retain their existing ownership.
    pd.testing.assert_frame_equal(best, _rescue_unassigned_nodes(best, pixels, nodes))


def test_coastal_rescue_fails_instead_of_stealing_last_land_coverage():
    pixels = gpd.GeoDataFrame(
        {"site_position": [0, 1]}, geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)], crs=3035,
    )
    nodes = gpd.GeoDataFrame(
        {"node_position": [0, 1], "node_id": ["coast", "single"], "country_id": ["AA"] * 2},
        geometry=[Point(0, 0), Point(1, 0)], crs=3035,
    )
    best = pd.DataFrame({"site_position": [0, 1], "node_position": [0, 1],
                         "node_id": ["coast", "single"], "country_id": ["AA"] * 2})
    with pytest.raises(RuntimeError, match="spare usable coverage"):
        _rescue_unassigned_nodes(best, pixels, nodes, onshore_mask=[False, True])
