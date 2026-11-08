import numpy as np
import pandas as pd
import pytest

import eurotracer as et


@pytest.fixture(scope="session")
def study(tmp_path_factory):
    root = tmp_path_factory.mktemp("study")
    source = root / "inputs" / "input1"
    source.mkdir(parents=True)
    pd.DataFrame(
        {
            "name": ["a", "b", "c", "d"],
            "x": [0, 1, 2, 3],
            "y": [50, 51, 50, 51],
            "country": ["DK", "DK", "NO", "NO"],
        }
    ).to_csv(source / "bus.csv", index=False)
    pd.DataFrame(
        {
            "link_id": ["ab", "bc", "cd", "da", "ac"],
            "source_node_id": ["a", "b", "c", "d", "a"],
            "destination_node_id": ["b", "c", "d", "a", "c"],
        }
    ).to_csv(source / "links.csv", index=False)
    time = pd.date_range("2013-01-01", periods=8760, freq="h")
    base = 1 + 0.15 * np.cos(np.arange(8760) * 2 * np.pi / 24)
    pd.DataFrame(
        {
            "time": time,
            **{node: base * demand for node, demand in zip("abcd", [10, 15, 12, 13], strict=True)},
        }
    ).to_csv(source / "load_template.csv", index=False)
    sites = pd.DataFrame(
        {
            "site_id": [f"p{i}" for i in range(10)],
            "node_id": [*"aabbccdd", "", ""],
            "longitude": [0, 0.1, 1, 1.1, 2, 2.1, 3, 3.1, 1.5, 1.6],
            "latitude": [50] * 10,
            "cell_area_km2": [10.0] * 10,
            "land_wind_area_km2": [10.0] * 8 + [0.0, 0.0],
            "solar_area_km2": [10.0] * 8 + [0.0, 0.0],
            "is_sea": [False] * 8 + [True, True],
            "water_depth_m": [0.0] * 8 + [20.0, 25.0],
            "distance_to_land_m": [0.0] * 8 + [2000.0, 2200.0],
            "basin_id": [""] * 8 + ["north", "north"],
        }
    )
    years = {}
    for year, hours in ((2019, 8760), (2020, 8784)):
        t = np.arange(hours)
        wind = np.stack(
            [
                np.clip(0.35 + 0.2 * np.cos(t * 2 * np.pi / (24 * (2 + i / 3)) + i), 0.02, 1)
                for i in range(10)
            ],
            axis=1,
        )
        solar = np.stack(
            [
                np.maximum(0, np.sin((t % 24 - 6) * np.pi / 12)) * (0.7 - i * 0.02)
                for i in range(10)
            ],
            axis=1,
        )
        years[year] = (wind, solar)
    archive = et.import_weather(
        root / "weather", sites, years, provenance={"fixture": "deterministic synthetic"}
    )
    rows = []
    for name, investment, unit in (
        ("onwind", 1000, "EUR/kW"),
        ("offwind", 3000, "EUR/kW"),
        ("solar-utility", 600, "EUR/kW"),
        ("CCGT", 800, "EUR/kW"),
        ("HVAC overhead", 400, "EUR/MW/km"),
        ("HVDC submarine", 600, "EUR/MW/km"),
        ("HVDC inverter pair", 150000, "EUR/MW"),
    ):
        for p, v, u in (
            ("investment", investment, unit),
            ("FOM", 2, "%/year"),
            ("lifetime", 25, "years"),
        ):
            rows.append((name, p, v, u))
    rows.extend([("CCGT", "efficiency", 0.5, "per unit"), ("gas", "fuel", 25, "EUR/MWh_th")])
    pd.DataFrame(rows, columns=["technology", "parameter", "value", "unit"]).to_csv(
        source / "costs_2030.csv", index=False
    )
    return root, archive


@pytest.fixture
def options(study):
    root, archive = study
    return dict(
        project=root, archive=archive, start="2019-12-31", stop="2020-01-03", progress=False
    )
