"""Proportional sharing, postage stamp, and capacity attribution.

The dense node-by-node color matrix lives for one hour only. Tail integration
weights are computed before tracing, eliminating the node-link-bin tensor.
"""

from __future__ import annotations

import heapq

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve_triangular

from . import storage as st


def technology_shares(wind, solar, backup, load):
    renewable = wind + solar
    fractions = np.divide(
        np.stack((wind, solar)), renewable, out=np.zeros((2, *renewable.shape)), where=renewable > 0
    )
    surplus = np.maximum(renewable - load, 0)
    mixed = (surplus > 0) & (backup > 0)
    renewable_fraction = np.where(surplus > 0, 1.0, 0.0)
    renewable_fraction[mixed] = surplus[mixed] / (surplus[mixed] + backup[mixed])
    backup_fraction = np.where(backup > 0, 1 - renewable_fraction, 0.0)
    return np.concatenate((fractions * renewable_fraction, backup_fraction[None]), axis=0)


def colors(injection, flow, source, destination):
    """Solve A Q = diag(P+) and A.T Qrev = diag(P-) in DAG order."""
    p = np.asarray(injection, dtype=float)
    f = np.asarray(flow, dtype=float)
    if not np.isfinite(p).all() or not np.isfinite(f).all():
        raise ValueError("Nonfinite tracing input")
    s = np.where(f >= 0, source, destination).astype(int)
    d = np.where(f >= 0, destination, source).astype(int)
    magnitude = np.abs(f)
    n = len(p)
    active = magnitude > 0
    adjacency: list[list[int]] = [[] for _ in range(n)]
    degree = np.zeros(n, dtype=int)
    for a, b in zip(s[active], d[active], strict=True):
        adjacency[a].append(b)
        degree[b] += 1
    ready = list(np.flatnonzero(degree == 0))
    heapq.heapify(ready)
    order = []
    while ready:
        a = heapq.heappop(ready)
        order.append(a)
        for b in adjacency[a]:
            degree[b] -= 1
            if degree[b] == 0:
                heapq.heappush(ready, int(b))
    if len(order) != n:
        raise ValueError("Directed flow contains a cycle; expected positive-susceptance DC flows")
    order = np.asarray(order)
    positive = np.maximum(p, 0)
    negative = np.maximum(-p, 0)
    diagonal = negative + np.bincount(s, weights=magnitude, minlength=n)
    inactive = diagonal == 0
    diagonal[inactive] = 1
    matrix = (diags(diagonal) - coo_matrix((magnitude, (d, s)), shape=(n, n))).tocsr()
    triangular = matrix[order][:, order]
    sources = np.flatnonzero(positive > 0)
    sinks = np.flatnonzero(negative > 0)
    results = []
    for values, indices, reverse in ((positive, sources, False), (negative, sinks, True)):
        result = np.zeros((n, len(indices)))
        if len(indices):
            rhs = np.zeros_like(result)
            rhs[indices, np.arange(len(indices))] = values[indices]
            solved = spsolve_triangular(
                triangular.T.tocsr() if reverse else triangular, rhs[order], lower=not reverse
            )
            result[order] = solved
            if (result < -1e-8).any() or not np.isfinite(result).all():
                raise ArithmeticError("Invalid proportional-sharing solution")
        results.append(result)
    return sources, sinks, results[0], results[1], s, d


def histogram_weights(counts, capacities):
    """Each eligible hour receives the integral coefficient of its loading bin.

    Right-continuous bins include f=K in the last bin. Above-K and zero flows
    are excluded. An empty upper interval uses the last observed tail mixture
    (its left limit); no discarded peaks are brought back into the estimate.
    """
    counts = np.asarray(counts, dtype=np.int64)
    capacities = np.asarray(capacities, dtype=float)
    tails = counts[:, ::-1].cumsum(axis=1)[:, ::-1]
    inverse = np.divide(1.0, tails, out=np.zeros_like(tails, dtype=float), where=tails > 0)
    delta = capacities / counts.shape[1]
    weights = delta[:, None] * inverse.cumsum(axis=1)
    for link in range(len(capacities)):
        occupied = np.flatnonzero(counts[link])
        if not len(occupied):
            if capacities[link] > 0:
                raise ValueError(
                    f"Link {link} has positive sized capacity but no eligible positive loading"
                )
            continue
        last = occupied[-1]
        weights[link, last] += delta[link] * (counts.shape[1] - 1 - last) / counts[link, last]
    return weights


def loading_bins(flow, capacity, bins):
    magnitude = np.abs(flow)
    valid = (magnitude > 0) & (magnitude <= capacity) & (capacity > 0)
    normalized = np.divide(magnitude, capacity, out=np.zeros_like(magnitude), where=capacity > 0)
    index = np.minimum((np.clip(normalized, 0, 1) * bins).astype(int), bins - 1)
    return index, valid


def generation_capacity(installed, exports, produced, curtailed):
    """Source rows close to installed capacity; sinks carry curtailed overhead."""
    net = produced - curtailed
    exchange = np.asarray(exports, dtype=float).copy()
    np.fill_diagonal(exchange, 0)
    if (exchange.sum(axis=1) > np.maximum(net, 0) + 1e-7 * np.maximum(1, produced)).any():
        raise ArithmeticError("Exported energy exceeds non-curtailed source generation")
    result = np.divide(exchange, net[:, None], out=np.zeros_like(exchange), where=net[:, None] > 0)
    result *= installed[:, None]
    result[np.diag_indices_from(result)] = installed - result.sum(axis=1)
    if (result < -1e-7 * np.maximum(1, installed[:, None])).any():
        raise ArithmeticError("Negative attributed generation capacity")
    return np.maximum(result, 0)


def energy_exchange(exports, produced):
    result = exports.copy()
    np.fill_diagonal(result, 0)
    residual = produced - result.sum(axis=1)
    if (residual < -1e-7 * np.maximum(produced, 1)).any():
        raise ArithmeticError("Attributed backup export exceeds generation")
    result[np.diag_indices_from(result)] = np.maximum(residual, 0)
    return result


def trace(run, method, bins=50, *, progress=True):
    if method not in ("flow_traced", "postage_stamp", "load_based"):
        raise ValueError("method must be flow_traced, postage_stamp or load_based")
    if not isinstance(bins, int) or bins < 2:
        raise ValueError("bins must be an integer >= 2")
    definition = {
        "operation": run.meta["operations"],
        "flow": run.meta["flow"],
        "method": method,
        "bins": bins,
        "algorithm": 5,
    }
    path = run.project / ".eurotracer" / "allocation" / st.identity(definition)
    with st.lock(path):
        if (path / "complete.json").exists():
            if progress:
                print(f"Reusing {method} attribution ({bins} loading bins)")
            return path
        data = run._data()
        nodes, links = run.nodes, run.links
        n, link_count = len(nodes), len(links)
        estimated = 8 * (12 * n * n + 4 * n * link_count + 10 * 168 * n)
        if estimated > run.meta["memory_gib"] * 1024**3 * 0.7:
            raise MemoryError(
                f"Tracing workspace requires at least {estimated / 1024**3 / 0.7:.2f} GiB"
            )
        op = st.group(run.operation_path / "arrays.zarr")
        flows = st.group(run.flow_path / "arrays.zarr")["flow"]
        capacity = links.capacity_mw.to_numpy()
        s, d = links.source.to_numpy(), links.destination.to_numpy()
        histfile = path / "histogram.npz"
        if method == "flow_traced":
            if histfile.exists():
                counts = st.load_arrays(histfile)["counts"]
            else:
                counts = np.zeros((link_count, bins), dtype=np.int64)
                for block in st.slices(len(data.time)):
                    index, valid = loading_bins(flows[block], capacity, bins)
                    rows, columns = np.nonzero(valid)
                    np.add.at(counts, (columns, index[rows, columns]), 1)
                st.save_arrays(histfile, counts=counts)
            weights = histogram_weights(counts, capacity)
        else:
            weights = np.zeros((link_count, bins))
        checkpoint = path / "checkpoint.npz"
        if checkpoint.exists():
            state = st.load_arrays(checkpoint)
        else:
            state = {
                "exports": np.zeros((3, n, n)),
                "produced": np.zeros((3, n)),
                "curtailed": np.zeros((3, n)),
                "load": np.zeros(n),
                "positive": np.zeros(n),
                "negative": np.zeros(n),
                "forward": np.zeros((n, link_count)),
                "reverse": np.zeros((n, link_count)),
                "hours": np.asarray(0, dtype=np.int64),
            }
        groups = (
            pd.factorize(nodes.country, sort=True)[0]
            if run.scenario.balancing == "country"
            else np.zeros(n, int)
        )
        pools = [np.flatnonzero(groups == g) for g in np.unique(groups)]
        done = int(state["hours"])
        if progress:
            print(f"{method}: attributing energy and capacity ({done:,} hours already committed)")
        last_year = None
        for block in st.slices(len(data.time), 168):
            if block.stop <= done:
                continue
            g, b = op["generation"][block], op["balancing"][block]
            load = np.pad(data.load(block), ((0, 0), (0, n - len(data.nodes))))
            p = g.sum(axis=2) - load - b
            backup = np.maximum(-b, 0)
            shares = technology_shares(g[:, :, 0], g[:, :, 1], backup, load)
            produced = np.stack((g[:, :, 0], g[:, :, 1], backup))
            # Imported curtailment stays with the importing sink. Only local,
            # non-importing curtailment reduces the source-output denominator.
            local_curtailment = np.where(p >= 0, np.maximum(b, 0), 0)
            ratio = np.divide(
                g,
                g.sum(axis=2)[:, :, None],
                out=np.zeros_like(g),
                where=g.sum(axis=2)[:, :, None] > 0,
            )
            state["curtailed"][:2] += (local_curtailment[:, :, None] * ratio).sum(axis=0).T
            state["produced"] += produced.sum(axis=1)
            state["load"] += load.sum(axis=0)
            state["positive"] += np.maximum(p, 0).sum(axis=0)
            state["negative"] += np.maximum(-p, 0).sum(axis=0)
            if method == "postage_stamp":
                plus, minus = np.maximum(p, 0), np.maximum(-p, 0)
                for pool in pools:
                    demand = minus[:, pool]
                    demand = np.divide(
                        demand,
                        demand.sum(axis=1)[:, None],
                        out=np.zeros_like(demand),
                        where=demand.sum(axis=1)[:, None] > 0,
                    )
                    for k in range(3):
                        exchange = (shares[k, :, pool].T * plus[:, pool]).T @ demand
                        state["exports"][k][np.ix_(pool, pool)] += exchange
            elif method == "flow_traced":
                flow = flows[block]
                index, valid = loading_bins(flow, capacity, bins)
                for h in range(len(p)):
                    sources, sinks, forward, reverse, send, receive = colors(p[h], flow[h], s, d)
                    exchange = forward.T * np.maximum(-p[h], 0)[None, :]
                    for k in range(3):
                        state["exports"][k, sources] += exchange * shares[k, h, sources, None]
                    selected = np.flatnonzero(valid[h])
                    weight = weights[selected, index[h, selected]]
                    state["forward"][np.ix_(sources, selected)] += (
                        forward[send[selected]].T * weight
                    )
                    state["reverse"][np.ix_(sinks, selected)] += (
                        reverse[receive[selected]].T * weight
                    )
            state["hours"] = np.asarray(block.stop, dtype=np.int64)
            if block.stop == len(data.time) or block.stop // 672 > block.start // 672:
                st.save_arrays(checkpoint, **state)
            year = data.time[block.start].year
            if progress and year != last_year:
                print(f"  {method} {year}: {block.stop:,}/{len(data.time):,} hours")
                last_year = year
        installed = np.vstack(
            (
                run._layout()["installed"],
                st.load_arrays(run.operation_path / "totals.npz")["backup_capacity"],
            )
        )
        if method == "load_based":
            load_share = state["load"] / state["load"].sum()
            generation = np.stack([np.diag(capacity) for capacity in installed])
            backup_energy = np.diag(state["produced"][2])
            state["forward"][:] = state["reverse"][:] = load_share[:, None] * capacity
        else:
            generation = np.stack(
                [
                    generation_capacity(
                        installed[k],
                        state["exports"][k],
                        state["produced"][k],
                        state["curtailed"][k],
                    )
                    for k in range(3)
                ]
            )
            backup_energy = energy_exchange(state["exports"][2], state["produced"][2])
            if method == "postage_stamp":
                for name, power in (("forward", "positive"), ("reverse", "negative")):
                    total = state[power].sum()
                    if total <= 0 and np.any(capacity > 0):
                        raise ArithmeticError("Positive link capacities with no power exchange")
                    state[name] = state[power][:, None] / max(total, 1e-30) * capacity
        for direction in ("forward", "reverse"):
            if not np.allclose(state[direction].sum(axis=0), capacity, rtol=1e-6, atol=1e-6):
                raise ArithmeticError(
                    f"{direction} transmission attribution does not conserve capacity"
                )
        st.save_arrays(
            path / "allocation.npz",
            generation=generation,
            backup_energy=backup_energy,
            exports=state["exports"] / len(data.time),
            forward=state["forward"],
            reverse=state["reverse"],
            load=state["load"],
            produced=state["produced"],
            curtailed=state["curtailed"],
        )
        st.write_json(
            path / "complete.json",
            {
                **definition,
                "generation_capacity_residual_mw": float(
                    np.abs(generation.sum(axis=2) - installed).max(initial=0)
                ),
                "link_capacity_residual_mw": float(
                    max(
                        np.abs(state[k].sum(axis=0) - capacity).max(initial=0)
                        for k in ("forward", "reverse")
                    )
                ),
                "backup_energy_residual_mwh": float(
                    abs(backup_energy.sum() - state["produced"][2].sum())
                ),
            },
        )
    return path
