"""Focused plots from published arrays and tables; no hidden model calculations."""

from __future__ import annotations

import numpy as np


def plot(result, kind="layout", *, start=None, stop=None, nodes=None, technology="wind", ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        figure, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    else:
        figure = ax.figure
    run = getattr(result, "run", result)
    if kind == "layout":
        table = run.layout
        if technology not in ("wind", "solar", "backup"):
            raise ValueError("technology must be wind, solar or backup")
        m = ax.scatter(
            table.longitude, table.latitude, c=table[f"{technology}_mw"], s=20, cmap="viridis"
        )
        figure.colorbar(m, ax=ax, label=f"{technology} capacity (MW)")
        ax.set(xlabel="Longitude", ylabel="Latitude", title=f"{run.name}: installed capacity")
    elif kind == "network":
        table = run.nodes
        for link in run.links.itertuples():
            pair = table.iloc[[link.source, link.destination]]
            ax.plot(
                pair.longitude,
                pair.latitude,
                color="tab:orange" if link.is_hvdc else "tab:blue",
                linewidth=0.5 + 2 * link.capacity_mw / max(run.links.capacity_mw.max(), 1),
            )
        ax.set(xlabel="Longitude", ylabel="Latitude", title=f"{run.name}: transmission")
    elif kind == "series":
        if start is None or stop is None:
            raise ValueError("Time-series plots require start and exclusive stop")
        values = run.series("generation", start=start, stop=stop, nodes=nodes).sum("node")
        for tech in values.renewable.values:
            ax.plot(values.time, values.sel(renewable=tech), label=str(tech))
        ax.plot(
            values.time,
            run.series("load", start=start, stop=stop, nodes=nodes).sum("node"),
            label="load",
            color="black",
        )
        ax.legend()
        ax.set(ylabel="MW", title=run.name)
    elif kind == "costs":
        from .economics import COMPONENTS

        table = result.countries
        bottom = np.zeros(len(table))
        for c in COMPONENTS:
            value = table[f"{c}_eur_per_mwh"].fillna(0).to_numpy()
            ax.bar(table.country, value, bottom=bottom, label=c.replace("_", " "))
            bottom += value
        ax.set(ylabel="EUR/MWh", title=f"{run.name}: {result.meta['method']}")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=90)
    elif kind == "exchanges":
        a = result.arrays.average_export.sel(technology=technology)
        countries = run.nodes.country.to_numpy()
        labels = sorted(set(countries))
        aggregation = np.array([countries == c for c in labels], float)
        values = aggregation @ a.to_numpy() @ aggregation.T
        m = ax.imshow(values, cmap="viridis")
        ax.set_xticks(range(len(labels)), labels, rotation=90)
        ax.set_yticks(range(len(labels)), labels)
        ax.set(
            xlabel="Importing country",
            ylabel="Exporting country",
            title=f"{technology} average export",
        )
        figure.colorbar(m, ax=ax, label="MW")
    else:
        raise ValueError("Plot kind must be layout, network, series, costs or exchanges")
    return figure


def comparison(table, *, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        figure, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    else:
        figure = ax.figure
    for key, frame in table.groupby("comparison"):
        ax.plot(
            frame.country,
            frame.total_eur_per_mwh,
            marker="o",
            label=f"{frame.run.iloc[0]} / {frame.method.iloc[0]} ({key})",
        )
    ax.set(ylabel="EUR/MWh", title="Country costs")
    ax.tick_params(axis="x", rotation=90)
    ax.legend()
    return figure
