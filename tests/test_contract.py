"""Boundary and integration checks independent of historical numerical outputs."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import eurotracer as et
from eurotracer import storage as st
from eurotracer.allocation import histogram_weights, loading_bins
from eurotracer.data import load_template, read_network
from eurotracer.economics import Prices, tables, validate_prices
from eurotracer.network import Network, build_links


def test_country_lcoe_is_ratio_of_sums():
    nodes = pd.DataFrame({"node_id": ["a", "b", "c"], "country": ["A", "A", "B"]})
    costs = np.zeros((3, 8))
    costs[:, 0] = [100, 300, 50]
    _, country = tables(nodes, costs, np.array([1.0, 9.0, 2.0]))
    assert country.total_eur_per_mwh.tolist() == [40.0, 25.0]


def test_tail_direct_integral_and_bin_convergence():
    flow = (np.arange(10000) + 0.5) / 10000
    # Two independent colors, with a known smooth limiting tail integral 3/4.
    errors = []
    for bins in (50, 200, 1000):
        index, valid = loading_bins(flow[:, None], np.ones(1), bins)
        counts = np.bincount(index[valid], minlength=bins)[None]
        weighted = flow @ histogram_weights(counts, [1.0])[0, index[:, 0]]
        # Independent bin-by-bin integration of actual tail mixtures.
        direct = sum(flow[index[:, 0] >= k].mean() / bins for k in range(bins))
        np.testing.assert_allclose(weighted, direct, atol=1e-13)
        errors.append(abs(weighted - 0.75))
    assert errors[2] < errors[1] < errors[0] < 0.006


def test_load_header_ids_are_reordered(tmp_path):
    table = pd.DataFrame({"b": np.full(8760, 2.0), "a": np.ones(8760)})
    file = tmp_path / "load.csv"
    table.to_csv(file, index=False)
    result = load_template(file, ["a", "b"])
    np.testing.assert_array_equal(result.iloc[0].to_numpy(), [1, 2])


def test_real_network_has_33_connected_domestic_components():
    source = Path(__file__).resolve().parents[1] / "Eurotracer_run/inputs/input1"
    if not source.exists():
        pytest.skip("Local thesis input not distributed with package")
    nodes, base = read_network(source)
    links = build_links(nodes, base, et.Scenario("LLnoT-C"))
    network = Network(len(nodes), links.source, links.destination, np.ones(len(links)))
    assert len(nodes) == 512 and len(base) == 956 and len(network.parts) == 33
    for positions, _, _ in network.parts:
        assert nodes.iloc[positions].country.nunique() == 1


def test_import_preserves_native_precision_and_rejects_changed_values(tmp_path):
    sites = pd.DataFrame({"site_id": ["x"]})
    x = np.full((8760, 1), 0.123456789012345, dtype="f8")
    path = et.import_weather(tmp_path / "cf", sites, {2013: (x, x)})
    stored = st.group(path / "years/2013/arrays.zarr")["wind"]
    assert stored.dtype == x.dtype and stored[0, 0] == x[0, 0]
    x[0] += 1e-10
    with pytest.raises(ValueError, match="Changed CF"):
        et.import_weather(path, sites, {2013: (x, x)})
    assert stored[0, 0] != x[0, 0]


def test_exact_columnwise_quantiles_preserve_double_precision(tmp_path):
    rng = np.random.default_rng(52)
    values = rng.normal(size=(1007, 7)) + np.arange(7) * 1e-10
    root = st.group(tmp_path / "quantiles.zarr", "a")
    saved = st.array(root, "values", values.shape)
    saved[:] = values
    np.testing.assert_array_equal(saved[:], values)
    np.testing.assert_array_equal(
        st.quantile_columns(saved, 0.99, absolute=True, memory_gib=0.001),
        np.quantile(abs(values), 0.99, axis=0, method="linear"),
    )
    np.testing.assert_array_equal(
        st.quantile_columns(saved, 0.99, negative=True, memory_gib=0.001),
        np.quantile(np.maximum(-values, 0), 0.99, axis=0, method="linear"),
    )


def test_cost_free_results_remain_useful_and_cost_optimization_fails(options):
    root = Path(options["project"])
    file = root / "inputs/input1/costs_2030.csv"
    original = file.read_bytes()
    try:
        file.unlink()
        run = et.run(**options, scenario=et.Scenario(alpha=0.41), name="no-costs")
        result = run.evaluate(method="postage_stamp", split=0.5, progress=False)
        assert "onshore_wind_mw" in result.countries
        assert "total_eur_per_mwh" not in result.countries
        with pytest.raises(ValueError, match="requires costs"):
            run.evaluate(progress=False)
        with pytest.raises(ValueError, match="requires valid costs"):
            et.compare(result)
    finally:
        file.write_bytes(original)


def test_load_benchmark_keeps_source_generation(options):
    run = et.run(**options, scenario=et.Scenario(alpha=0.6), name="load-benchmark")
    result = run.evaluate(method="load_based", split=0.5, progress=False)
    np.testing.assert_allclose(
        result.nodes.total_eur_per_year, result.installed_nodes.total_eur_per_year
    )
    assert result.arrays.generation_capacity.sum("sink_node").shape == (3, 4)


def test_postage_stamp_uses_country_pools_for_ll(options):
    run = et.run(**options, scenario=et.Scenario("LL-C", alpha=0.5), name="country-pools")
    result = run.evaluate(method="postage_stamp", split=0.5, progress=False)
    exchange = result.arrays.average_export.to_numpy()
    countries = run.nodes.country.to_numpy()
    np.testing.assert_array_equal(exchange[:, countries[:, None] != countries[None, :]], 0)


def test_density_topology_reuse_and_length_weighting(options):
    a = et.run(**options, scenario=et.Scenario(alpha=0.5), name="selective")
    b = et.run(**options, scenario=et.Scenario(alpha=0.5, length_exponent=1.0), name="selective")
    c = et.run(
        **options, scenario=et.Scenario(alpha=0.5, wind_density_mw_per_km2=8.0), name="selective"
    )
    assert a.operation_path == b.operation_path and a.flow_path != b.flow_path
    assert a.layout_path != c.layout_path
    assert (
        st.read_json(a.layout_path / "complete.json")["basis_path"]
        == st.read_json(c.layout_path / "complete.json")["basis_path"]
    )


def test_trace_checkpoint_restart_matches_completed_result(options, monkeypatch):
    from eurotracer import allocation

    run = et.run(**options, scenario=et.Scenario(alpha=0.523), name="trace-restart")
    original = allocation.st.save_arrays

    def interrupted(path, **arrays):
        original(path, **arrays)
        if Path(path).name == "checkpoint.npz":
            raise RuntimeError("interrupted after committed checkpoint")

    with monkeypatch.context() as patch:
        patch.setattr(allocation.st, "save_arrays", interrupted)
        with pytest.raises(RuntimeError, match="committed"):
            run.evaluate(split=0.5, progress=False)
    result = run.evaluate(split=0.5, progress=False)
    np.testing.assert_allclose(
        result.nodes.total_eur_per_year.sum(), result.installed_nodes.total_eur_per_year.sum()
    )
    np.testing.assert_allclose(
        result.arrays.generation_capacity.sum("sink_node")[0], run.layout.wind_mw
    )


def test_actual_cost_table_units_are_accepted(options):
    cost = Path(__file__).resolve().parents[1] / "Eurotracer_run/inputs/input1/costs_2030.csv"
    if not cost.exists():
        pytest.skip("Local thesis cost inputs not distributed with package")
    run = et.run(**options, scenario=et.Scenario(alpha=0.6), name="real-prices")
    installed = run.layout[["wind_mw", "solar_mw", "backup_mw"]].to_numpy().T
    validate_prices(Prices(pd.read_csv(cost)), run.nodes, run.links, installed)
    result = run.evaluate(split="minimize_wsd", cost_table=cost, progress=False)
    assert np.isfinite(result.countries.total_eur_per_mwh).all()


def test_plots_read_results_without_recomputation(options, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from eurotracer import allocation, plotting

    run = et.run(**options, scenario=et.Scenario(alpha=0.6), name="plotting")
    result = run.evaluate(split=0.5, progress=False)
    monkeypatch.setattr(allocation, "trace", lambda *args, **kwargs: pytest.fail("hidden tracing"))
    for kind in ("layout", "network"):
        plt.close(run.plot(kind))
    plt.close(run.plot("series", start="2020-01-01", stop="2020-01-02"))
    for kind in ("costs", "exchanges"):
        plt.close(result.plot(kind))
    plt.close(plotting.comparison(et.compare(result)))
