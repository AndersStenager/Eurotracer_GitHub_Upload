"""Run a four-node, two-country example using only generated synthetic inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import eurotracer as et


def make_inputs(project: Path) -> Path:
    """Create a deterministic toy network, annual resource arrays and toy prices."""
    source = project / "inputs" / "input1"
    source.mkdir(parents=True, exist_ok=True)
    nodes = ["a", "b", "c", "d"]
    pd.DataFrame(
        {
            "name": nodes,
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [50.0, 51.0, 50.0, 51.0],
            "country": ["AA", "AA", "BB", "BB"],
        }
    ).to_csv(source / "bus.csv", index=False)
    pd.DataFrame(
        {
            "link_id": ["ab", "bc", "cd", "da", "ac"],
            "source_node_id": ["a", "b", "c", "d", "a"],
            "destination_node_id": ["b", "c", "d", "a", "c"],
        }
    ).to_csv(source / "links.csv", index=False)

    # The public importer requires a whole year; the model will use only 72 hours.
    time = pd.date_range("2019-01-01", periods=8760, freq="h")
    hours = np.arange(len(time))
    daily_load = 1 + 0.15 * np.cos(hours * 2 * np.pi / 24)
    pd.DataFrame(
        {
            "time": time,
            **{
                node: daily_load * demand
                for node, demand in zip(nodes, [10.0, 15.0, 12.0, 13.0], strict=True)
            },
        }
    ).to_csv(source / "load_template.csv", index=False)
    sites = pd.DataFrame(
        {
            "site_id": ["site-a", "site-b", "site-c", "site-d"],
            "node_id": nodes,
            "longitude": [0.0, 1.0, 2.0, 3.0],
            "latitude": [50.0, 51.0, 50.0, 51.0],
            "cell_area_km2": [10.0] * 4,
            "land_wind_area_km2": [10.0] * 4,
            "solar_area_km2": [10.0] * 4,
            "is_sea": [False] * 4,
        }
    )
    wind = np.stack(
        [0.35 + 0.2 * np.cos(hours * 2 * np.pi / (24 * (2 + i / 3)) + i) for i in range(4)],
        axis=1,
    )
    solar = np.stack(
        [np.maximum(0, np.sin((hours % 24 - 6) * np.pi / 12)) * (0.7 - i * 0.02) for i in range(4)],
        axis=1,
    )
    archive = et.import_weather(
        project / "weather",
        sites,
        {2019: (wind, solar)},
        provenance={"example": "deterministic synthetic portfolio demo"},
    )

    # Illustrative prices, not market estimates or thesis assumptions.
    rows = []
    for technology, investment, unit in (
        ("onwind", 1000, "EUR/kW"),
        ("solar-utility", 600, "EUR/kW"),
        ("CCGT", 800, "EUR/kW"),
        ("HVAC overhead", 400, "EUR/MW/km"),
        ("HVDC submarine", 600, "EUR/MW/km"),
        ("HVDC inverter pair", 150000, "EUR/MW"),
    ):
        for parameter, value, parameter_unit in (
            ("investment", investment, unit),
            ("FOM", 2.0, "%/year"),
            ("lifetime", 25.0, "years"),
        ):
            rows.append((technology, parameter, value, parameter_unit))
    rows.extend([("CCGT", "efficiency", 0.5, "per unit"), ("gas", "fuel", 25, "EUR/MWh_th")])
    pd.DataFrame(rows, columns=["technology", "parameter", "value", "unit"]).to_csv(
        source / "costs_2030.csv", index=False
    )
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("demo-output"))
    args = parser.parse_args()
    project = args.output.resolve()
    # Never overwrite an unrelated existing project with generated demo inputs.
    marker = project / "SYNTHETIC_DEMO.txt"
    if project.exists() and any(project.iterdir()) and not marker.exists():
        parser.error("Output is not an existing demo or an empty folder; choose another path.")
    project.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        "Synthetic four-node portfolio example. Not thesis or market results.\n",
        encoding="utf-8",
    )
    archive = make_inputs(project)
    run = et.run(
        project,
        et.Scenario("GG-C", alpha=0.6),
        start="2019-01-01",
        stop="2019-01-04",
        name="synthetic-demo",
        archive=archive,
        download=False,
        memory_gib=0.25,
        progress=False,
    )
    result = run.evaluate(method="flow_traced", split=0.5, progress=False)
    np.testing.assert_allclose(run.series("injection").to_numpy().sum(axis=1), 0.0, atol=1e-9)
    np.testing.assert_allclose(
        result.nodes.total_eur_per_year.sum(),
        result.installed_nodes.total_eur_per_year.sum(),
        rtol=1e-9,
    )
    columns = ["country", "demand_mwh", "backup_mwh", "total_eur_per_mwh"]
    countries = result.countries[columns]
    if len(countries) != 2 or not np.isfinite(countries[columns[1:]].to_numpy()).all():
        raise ArithmeticError("Expected two countries with finite results")
    countries.to_csv(project / "countries.csv", index=False)
    run.summary.to_csv(project / "run_summary.csv", index=False)
    print("SYNTHETIC EXAMPLE: four nodes, two fictional countries, 72 hours.")
    print("Illustrative inputs and costs only; these are not European market results.")
    print(countries.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Maximum nodal balance residual: {run.summary.max_balance_residual_mw.iloc[0]:.3g} MW")
    print("Energy balance and total attributed cost checks passed.")
    print(f"Country results: {project / 'countries.csv'}")


if __name__ == "__main__":
    main()
