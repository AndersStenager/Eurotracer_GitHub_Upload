from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import eurotracer as et
from eurotracer.data import expand_load, timeline


def test_four_scenarios_and_two_accounting_methods(options):
    runs = {}
    for preset in ("LLnoT-C", "LL-C", "LG-C", "GG-C"):
        run = et.run(**options, scenario=et.Scenario(preset), name=preset)
        runs[preset] = run
        for method in ("flow_traced", "postage_stamp"):
            result = run.evaluate(method=method, split="minimize_wsd", progress=False)
            assert np.isfinite(result.countries.total_eur_per_mwh).all()
            np.testing.assert_allclose(
                result.nodes.total_eur_per_year.sum(),
                result.installed_nodes.total_eur_per_year.sum(),
            )
            np.testing.assert_allclose(
                result.arrays.link_capacity.sum("sink_node"), run.links.capacity_mw, rtol=1e-6
            )
        assert run.summary.max_balance_residual_mw.iloc[0] < 1e-8
        reopened = et.open_run(run.path)
        assert reopened.id == run.id
        assert len(reopened.series("flow", start="2020-01-01", stop="2020-01-02")) == 24
        assert set(run.annual.year) == {2019, 2020}
    assert runs["LLnoT-C"].layout_path == runs["LL-C"].layout_path == runs["LG-C"].layout_path
    assert len(runs["LLnoT-C"].links) == 2
    assert len(runs["LL-C"].links) == 5
    assert len(runs["LL-C"].alpha) == 2


def test_postmodel_changes_reuse_physics_and_cost_conservation(options):
    run = et.run(**options, scenario=et.Scenario(alpha=0.6), name="cost-reuse")
    before = (run.operation_path / "complete.json").stat().st_mtime_ns
    a = run.evaluate(split=0.0, progress=False)
    b = run.evaluate(split=1.0, costs=et.CostAssumptions(rate=0.08), progress=False)
    assert before == (run.operation_path / "complete.json").stat().st_mtime_ns
    assert a.meta["allocation_path"] == b.meta["allocation_path"]
    assert a.path != b.path
    assert a.summary.eu_lcoe_eur_per_mwh < b.summary.eu_lcoe_eur_per_mwh
    assert len(et.compare(a, b)) == 4


def test_input_cost_edits_create_revision_without_locking_or_recomputing_physics(options):
    root = Path(options["project"])
    cost = root / "inputs/input1/costs_2030.csv"
    original = cost.read_bytes()
    try:
        a = et.run(**options, scenario=et.Scenario(alpha=0.5), name="revision")
        table = pd.read_csv(cost)
        table.loc[table.parameter == "investment", "value"] *= 1.1
        table.to_csv(cost, index=False)
        b = et.run(**options, scenario=et.Scenario(alpha=0.5), name="revision")
        assert a.id != b.id and a.operation_path == b.operation_path and a.flow_path == b.flow_path
        assert et.open_run(root, "revision").id == b.id
        assert et.open_run(root, "revision", revision=a.id).id == a.id
    finally:
        cost.write_bytes(original)


def test_restart_after_uncommitted_physical_block(options, monkeypatch):
    scenario = et.Scenario(alpha=0.4321)
    from eurotracer import operations

    original = operations.st.write_json
    failed = [False]

    def interrupt(path, value):
        if (
            Path(path).name == "progress.json"
            and "operations" in Path(path).parts
            and not failed[0]
        ):
            failed[0] = True
            raise RuntimeError("simulated process interruption")
        return original(path, value)

    with monkeypatch.context() as m:
        m.setattr(operations.st, "write_json", interrupt)
        with pytest.raises(RuntimeError, match="simulated"):
            et.run(**options, scenario=scenario, name="restart")
    run = et.run(**options, scenario=scenario, name="restart")
    injected = run.series("injection").to_numpy()
    np.testing.assert_allclose(injected.sum(axis=1), 0, atol=1e-10)
    assert run.series("generation").shape == (72, 4, 2)


def test_block_reader_never_reads_unrequested_weather_hours(options, monkeypatch):
    data = et.prepare(**options)
    import zarr

    original = zarr.Array.__getitem__
    reads = []

    def read(self, key):
        reads.append(key)
        return original(self, key)

    with monkeypatch.context() as m:
        m.setattr(zarr.Array, "__getitem__", read)
        wind, solar = data.cf(slice(22, 27))
    assert wind.shape == solar.shape == (5, 10)
    assert len(reads) == 4
    assert all(isinstance(k, slice) and k.stop - k.start <= 3 for k in reads)


def test_leap_day_and_half_open_calendar():
    template = pd.DataFrame(
        {"a": np.arange(8760.0)}, index=pd.date_range("2013-01-01", periods=8760, freq="h")
    )
    t = timeline(start="2020-02-28", stop="2020-03-02")
    values = expand_load(template, t)
    np.testing.assert_allclose(values[24:48], 0.5 * (values[:24] + values[48:]))
    assert len(timeline(years=(2020, 2020))) == 8784


@pytest.mark.parametrize("basis,scheme", [("pixel", "bus_avg"), ("pixel", "direct")])
def test_pixel_placement_preserves_energy_targets(options, basis, scheme):
    scenario = et.Scenario(alpha=0.6, basis=basis, pixel_scheme=scheme)
    run = et.run(**options, scenario=scenario, name=f"pixel-{scheme}")
    generation = run.series("generation").to_numpy().sum()
    demand = run.series("load").to_numpy().sum()
    np.testing.assert_allclose(generation, demand, rtol=2e-7)


def test_offshore_source_costs_and_balanced_hubs(options):
    offshore = et.Offshore(10.0, (et.Basin("north", 1.0, 1, 1, ("DK", "NO")),))
    run = et.run(**options, scenario=et.Scenario(alpha=0.6, offshore=offshore), name="offshore")
    assert len(run.nodes) == 5
    np.testing.assert_allclose(run.layout.loc[run.nodes.is_offshore, "wind_mw"].sum(), 10)
    result = run.evaluate(split=0.5, progress=False)
    np.testing.assert_allclose(
        result.nodes.offshore_wind_eur_per_year.sum(),
        result.installed_nodes.offshore_wind_eur_per_year.sum(),
    )
    assert result.nodes.loc[~run.nodes.is_offshore, "offshore_wind_eur_per_year"].sum() > 0


def test_reopened_run_does_not_read_weather_arrays(options, monkeypatch):
    run = et.run(**options, scenario=et.Scenario(alpha=0.6), name="lazy-open")
    import zarr

    with monkeypatch.context() as m:
        m.setattr(
            zarr.Array,
            "__getitem__",
            lambda *a: (_ for _ in ()).throw(AssertionError("array read")),
        )
        reopened = et.open_run(run.path)
        assert reopened.id == run.id
        assert reopened.summary.nodes.iloc[0] == 4
