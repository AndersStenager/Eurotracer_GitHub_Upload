"""Component-wise DC power flow with reusable sparse Laplacian factorizations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu


def distances(lon1, lat1, lon2, lat2):
    a, b, c, d = map(
        np.radians, (np.asarray(lon1), np.asarray(lat1), np.asarray(lon2), np.asarray(lat2))
    )
    v = np.sin((d - b) / 2) ** 2 + np.cos(b) * np.cos(d) * np.sin((c - a) / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(np.clip(v, 0, 1)))


def build_links(nodes, base, scenario):
    links = base.copy()
    ids = pd.Index(nodes.node_id)
    s, d = (ids.get_indexer(links[c]) for c in ("source_node_id", "destination_node_id"))
    if (s < 0).any() or (d < 0).any():
        raise ValueError("Unknown link endpoint")
    x, y = nodes.longitude.to_numpy(), nodes.latitude.to_numpy()
    links["length_km"] = distances(x[s], y[s], x[d], y[d])
    links["is_hvdc"] = False
    count = int(np.ceil(len(links) * scenario.hvdc_fraction))
    if count:
        links.loc[links.length_km.sort_values(kind="stable").index[-count:], "is_hvdc"] = True
    links["is_offshore"] = False
    if scenario.offshore is not None:
        additions = []
        for policy in scenario.offshore.basins:
            hubs = nodes.index[
                nodes.get("basin_id", pd.Series("", index=nodes.index)) == policy.basin_id
            ]
            for hub in hubs:
                eligible = np.flatnonzero(
                    (~nodes.is_offshore) & nodes.country.isin(policy.countries)
                )
                if scenario.domestic_only:
                    eligible = eligible[
                        nodes.country.iloc[eligible].to_numpy() == nodes.country.iloc[hub]
                    ]
                if len(eligible) < policy.landfalls:
                    raise ValueError(
                        f"Not enough permitted landfalls for {nodes.node_id.iloc[hub]}"
                    )
                length = distances(x[hub], y[hub], x[eligible], y[eligible])
                selected = np.lexsort((eligible, length))[: policy.landfalls]
                for rank in selected:
                    additions.append(
                        dict(
                            link_id=f"{nodes.node_id.iloc[hub]}-to-{nodes.node_id.iloc[eligible[rank]]}",
                            source_node_id=nodes.node_id.iloc[hub],
                            destination_node_id=nodes.node_id.iloc[eligible[rank]],
                            length_km=length[rank],
                            is_hvdc=True,
                            is_offshore=True,
                        )
                    )
        if additions:
            links = pd.concat([links, pd.DataFrame(additions)], ignore_index=True)
    s, d = (ids.get_indexer(links[c]) for c in ("source_node_id", "destination_node_id"))
    if scenario.domestic_only:
        keep = nodes.country.to_numpy()[s] == nodes.country.to_numpy()[d]
        links = links.loc[keep].reset_index(drop=True)
        s, d = s[keep], d[keep]
    links["source"] = s
    links["destination"] = d
    if (links.length_km <= 0).any():
        raise ValueError("Physical links must have positive length")
    return links


class Network:
    def __init__(self, count, source, destination, susceptance):
        self.count = count
        self.source = np.asarray(source, dtype=int)
        self.destination = np.asarray(destination, dtype=int)
        self.b = np.asarray(susceptance, dtype=float)
        if (
            self.source.shape != self.destination.shape
            or self.source.shape != self.b.shape
            or (self.b <= 0).any()
            or not np.isfinite(self.b).all()
        ):
            raise ValueError("Invalid network edge arrays")
        if (
            (self.source < 0).any()
            or (self.destination < 0).any()
            or (self.source >= count).any()
            or (self.destination >= count).any()
            or (self.source == self.destination).any()
        ):
            raise ValueError("Invalid network endpoint")
        edge = np.arange(len(self.b))
        self.incidence = coo_matrix(
            (
                np.r_[np.ones(len(edge)), -np.ones(len(edge))],
                (np.r_[self.source, self.destination], np.r_[edge, edge]),
            ),
            shape=(count, len(edge)),
        ).tocsc()
        graph = coo_matrix(
            (np.ones(len(edge)), (self.source, self.destination)), shape=(count, count)
        )
        n, self.component = connected_components(graph, directed=False)
        laplacian = self.incidence @ diags(self.b) @ self.incidence.T
        self.parts = []
        for c in range(n):
            nodes = np.flatnonzero(self.component == c)
            free = nodes[1:]
            factor = splu(laplacian[free][:, free].tocsc()) if len(free) else None
            self.parts.append((nodes, free, factor))

    def solve(self, injection):
        p = np.asarray(injection, dtype=float)
        if p.ndim != 2 or p.shape[1] != self.count or not np.isfinite(p).all():
            raise ValueError("Injection must be finite time-by-node data")
        theta = np.zeros_like(p)
        for nodes, free, factor in self.parts:
            residual = p[:, nodes].sum(axis=1)
            tolerance = 1e-9 * np.maximum(1, np.abs(p[:, nodes]).sum(axis=1))
            if (np.abs(residual) > tolerance).any():
                raise ValueError(
                    f"Unbalanced connected component containing node {nodes[0]}: "
                    f"{np.max(np.abs(residual)):.6g} MW"
                )
            if factor is not None:
                theta[:, free] = factor.solve(p[:, free].T).T
        flow = (theta[:, self.source] - theta[:, self.destination]) * self.b
        error = np.max(np.abs((self.incidence @ flow.T).T - p), initial=0.0)
        if error > 1e-7 * max(1, float(np.max(np.abs(p), initial=0))):
            raise ArithmeticError(f"DC nodal balance residual {error} MW")
        return flow, theta, float(error)


def balancing(generation, load, groups, mean_load):
    mismatch = generation.sum(axis=2) - load
    groups = np.asarray(groups, dtype=int)
    total = np.zeros((len(load), int(groups.max()) + 1))
    np.add.at(total.T, groups, mismatch.T)
    regional = np.bincount(groups, weights=mean_load, minlength=total.shape[1])
    if (regional <= 0).any():
        raise ValueError("Every balancing region must have positive mean demand")
    balance = total[:, groups] * (mean_load / regional[groups])
    return balance, mismatch - balance
