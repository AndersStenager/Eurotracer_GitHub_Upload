import numpy as np
import pandas as pd
import pytest

from eurotracer.allocation import (
    colors,
    generation_capacity,
    histogram_weights,
    loading_bins,
    technology_shares,
)
from eurotracer.economics import Prices, optimal_split, price_assets, weighted_dispersion
from eurotracer.network import Network
from eurotracer.placement import allocate


def test_link_colors_use_upstream_exporter_and_downstream_importer():
    sources, sinks, forward, reverse, send, receive = colors(
        np.array([1.0, 1.0, -2.0]), np.ones(2), np.array([0, 1]), np.array([2, 2])
    )
    np.testing.assert_array_equal(sources, [0, 1])
    np.testing.assert_allclose(forward[send].T, np.eye(2))
    np.testing.assert_allclose(reverse[receive], 1)
    # Reversing flow turns the example into one exporter and two consumers.
    sources, sinks, forward, reverse, send, receive = colors(
        np.array([-1.0, -1.0, 2.0]), -np.ones(2), np.array([0, 1]), np.array([2, 2])
    )
    np.testing.assert_allclose(reverse[receive].T, np.eye(2))
    np.testing.assert_allclose(forward[send], 1)


def test_sparse_tracing_matches_independent_dense_conservation_system():
    rng = np.random.default_rng(42)
    n = 12
    s = np.r_[np.arange(n - 1), [0, 1, 3, 4, 5]]
    d = np.r_[np.arange(1, n), [5, 7, 8, 10, 11]]
    network = Network(n, s, d, rng.uniform(0.5, 2, len(s)))
    p = rng.normal(size=(6, n))
    p -= p.mean(axis=1)[:, None]
    flows, _, _ = network.solve(p)
    for injection, flow in zip(p, flows, strict=True):
        upstream = np.where(flow > 0, s, d)
        downstream = np.where(flow > 0, d, s)
        a = np.diag(np.maximum(-injection, 0))
        for left, right, power in zip(upstream, downstream, abs(flow), strict=True):
            a[left, left] += power
            a[right, left] -= power
        expected = np.linalg.solve(a, np.diag(np.maximum(injection, 0)))
        expected_reverse = np.linalg.solve(a.T, np.diag(np.maximum(-injection, 0)))
        sources, sinks, q, r, _, _ = colors(injection, flow, s, d)
        np.testing.assert_allclose(q, expected[:, sources], atol=1e-10)
        np.testing.assert_allclose(r, expected_reverse[:, sinks], atol=1e-10)
        np.testing.assert_allclose(q.sum(axis=1), 1, atol=1e-10)


def test_domestic_islands_and_imbalance_rejection():
    model = Network(4, [0, 2], [1, 3], [1.0, 2.0])
    flow, _, error = model.solve(np.array([[3.0, -3.0, 4.0, -4.0]]))
    np.testing.assert_allclose(flow, [[3.0, 4.0]])
    assert error < 1e-10
    with pytest.raises(ValueError, match="Unbalanced connected component"):
        model.solve(np.array([[3.0, -2.0, 4.0, -5.0]]))


def test_generation_capacity_charges_curtailment_to_remaining_output():
    result = generation_capacity(
        np.array([100.0, 0]),
        np.array([[0.0, 60.0], [0, 0]]),
        np.array([100.0, 0]),
        np.array([20.0, 0]),
    )
    np.testing.assert_allclose(result, [[25.0, 75.0], [0, 0]])


def test_export_share_boundary_and_three_operating_states():
    shares = technology_shares(
        np.array([6.0, 1.0, 6.0, 5.0]),
        np.array([4.0, 1.0, 4.0, 0.0]),
        np.array([0.0, 7.0, 3.0, 2.0]),
        np.array([5.0, 5.0, 5.0, 5.0]),
    )
    np.testing.assert_allclose(shares[:, 0], [0.6, 0.4, 0])
    np.testing.assert_allclose(shares[:, 1], [0, 0, 1])
    np.testing.assert_allclose(shares[:, 2], [0.375, 0.25, 0.375])
    np.testing.assert_allclose(shares[:, 3], [0, 0, 1])


def test_tail_integral_excludes_peaks_and_conserves_capacity():
    flow = np.array([[0.0], [1.0], [2.0], [4.0], [5.0]])
    index, valid = loading_bins(flow, np.array([4.0]), 2)
    assert valid[:, 0].tolist() == [False, True, True, True, False]
    counts = np.array([[1, 2]])
    weights = histogram_weights(counts, [4.0])
    # Independent discrete integral: 0..2 uses all three hours; 2..4 uses two.
    np.testing.assert_allclose(weights, [[2 / 3, 2 / 3 + 1]])
    np.testing.assert_allclose((weights * counts).sum(axis=1), [4])
    np.testing.assert_allclose((histogram_weights([[2, 0, 0, 0]], [4]) * [[2, 0, 0, 0]]).sum(), 4)
    with pytest.raises(ValueError, match="no eligible positive loading"):
        histogram_weights([[0, 0]], [1.0])
    np.testing.assert_allclose(histogram_weights([[0, 0]], [0.0]), 0)


def test_capped_merit_and_infeasible_resource():
    result = allocate([0.2, 0.4], [100, 20], 20)
    np.testing.assert_allclose(result, [60, 20])
    np.testing.assert_allclose(result @ np.array([0.2, 0.4]), 20)
    with pytest.raises(ValueError, match="exceeds feasible"):
        allocate([0.2, 0.4], [100, 20], 30)


def test_analytic_country_split_matches_exhaustive_sweep():
    a = np.array([10.0, 40.0, 75.0])
    b = np.array([20.0, -10.0, -30.0])
    w = np.array([1.0, 3.0, 2.0])
    selected = optimal_split(a, b, w)
    grid = np.linspace(0, 1, 10001)
    brute = grid[np.argmin([weighted_dispersion(a + s * b, w) for s in grid])]
    assert abs(selected - brute) < 1e-4
    assert optimal_split(a, np.ones(3), w) == 0.5


def test_offshore_import_and_used_fuel_are_priced_at_source(study):
    root, _ = study
    prices = Prices(pd.read_csv(root / "inputs/input1/costs_2030.csv"))
    nodes = pd.DataFrame({"is_offshore": [True, False]})
    links = pd.DataFrame({"is_hvdc": pd.Series(dtype=bool), "capacity_mw": [], "length_km": []})
    installed = np.array([[100.0, 0.0], [0, 0], [10.0, 0]])
    generation = np.zeros((3, 2, 2))
    generation[0, 0, 1] = 100
    generation[2, 0, 1] = 10
    allocation = {
        "generation": generation,
        "load": np.array([0.0, 100.0]),
        "produced": np.array([[100.0, 0], [0, 0], [20.0, 0]]),
        "backup_energy": np.array([[0.0, 20.0], [0, 0]]),
        "forward": np.zeros((2, 0)),
        "reverse": np.zeros((2, 0)),
    }
    physical, used, _, _ = price_assets(nodes, links, installed, allocation, 1.0, prices)
    assert used[1, 1] == physical[0, 1] > 0
    assert used[1, 0] == 0
    assert used[1, 4] == physical[0, 4] == 1000
