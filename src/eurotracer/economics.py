"""Source-priced assets, used fuel, country LCOE and analytic fairness splits."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CostAssumptions

COMPONENTS = (
    "onshore_wind",
    "offshore_wind",
    "solar",
    "backup_capacity",
    "backup_fuel",
    "ac_transmission",
    "dc_transmission",
    "dc_converters",
)


def annuity(rate, lifetime):
    if not np.isfinite(rate) or rate < 0 or not np.isfinite(lifetime) or lifetime <= 0:
        raise ValueError("Invalid interest rate or asset lifetime")
    return 1 / lifetime if rate == 0 else rate / (1 - (1 + rate) ** (-lifetime))


class Prices:
    def __init__(self, table, assumptions=None):
        self.assumptions = assumptions or CostAssumptions()
        required = {"technology", "parameter", "value", "unit"}
        if not required.issubset(table) or table.duplicated(["technology", "parameter"]).any():
            raise ValueError("Costs require unique technology/parameter rows, value, unit")
        self.table = table.set_index(["technology", "parameter"])

    def get(self, technology, parameter, unit):
        try:
            row = self.table.loc[(technology, parameter)]
        except KeyError as exc:
            raise ValueError(f"Missing cost {technology}/{parameter}") from exc
        if str(row.unit).strip() not in ((unit,) if isinstance(unit, str) else unit):
            raise ValueError(f"Cost {technology}/{parameter}: expected {unit}, got {row.unit}")
        value = self.assumptions.overrides.get(technology, {}).get(parameter, float(row.value))
        if (
            not np.isfinite(value)
            or value < 0
            or (parameter in ("lifetime", "efficiency") and value == 0)
        ):
            raise ValueError(f"Invalid cost {technology}/{parameter}")
        if parameter == "efficiency" and value > 1:
            raise ValueError("Efficiency must be in (0,1]")
        return value

    def annual(self, technology, unit, scale=1.0):
        investment = self.get(technology, "investment", unit) * scale
        return investment * (
            annuity(self.assumptions.rate, self.get(technology, "lifetime", "years"))
            + self.get(technology, "FOM", "%/year") / 100
        )


def weighted_dispersion(values, weights):
    values, weights = np.asarray(values), np.asarray(weights)
    if (
        np.any(weights < 0)
        or not np.isfinite(values).all()
        or not np.isfinite(weights).all()
        or weights.sum() <= 0
    ):
        raise ValueError("WSD requires finite values and positive total nonnegative weight")
    weights = weights / weights.sum()
    return float(np.sqrt(weights @ ((values - weights @ values) ** 2)))


def validate_prices(prices, nodes, links, installed):
    """Fail before expensive attribution when an active asset cannot be priced."""
    offshore = nodes.is_offshore.to_numpy(bool)
    for technology, capacity in (
        ("onwind", installed[0, ~offshore]),
        ("offwind", installed[0, offshore]),
        ("solar-utility", installed[1]),
        ("CCGT", installed[2]),
    ):
        if np.any(capacity > 0):
            prices.annual(technology, ("EUR/kW", "EUR/kW_e", "EUR/kWel"), 1000)
    # Fuel may be used in rare hours even when Q99 backup capacity is zero.
    prices.get("gas", "fuel", "EUR/MWh_th")
    prices.get("CCGT", "efficiency", "per unit")
    for dc, technology in ((False, "HVAC overhead"), (True, "HVDC submarine")):
        if ((links.is_hvdc == dc) & (links.capacity_mw > 0)).any():
            if dc and (technology, "investment") not in prices.table.index:
                technology = "HVDC underground"
            prices.annual(technology, "EUR/MW/km")
            if dc:
                prices.annual("HVDC inverter pair", "EUR/MW")


def capacity_tables(nodes, links, installed, allocation, split, *, physical=False):
    """Published capacities and actual-period energy; no prices are required."""
    used = installed if physical else allocation["generation"].sum(axis=1)
    offshore = nodes.is_offshore.to_numpy(bool)
    wind = (
        installed[0] * offshore if physical else allocation["generation"][0, offshore].sum(axis=0)
    )
    grid = (
        allocation["load"][:, None] / allocation["load"].sum() * links.capacity_mw.to_numpy()
        if physical
        else (1 - split) * allocation["reverse"] + split * allocation["forward"]
    )
    fuel = allocation["produced"][2] if physical else allocation["backup_energy"].sum(axis=0)
    frame = nodes[["node_id", "country"]].copy()
    frame["onshore_wind_mw"] = used[0] - wind
    frame["offshore_wind_mw"] = wind
    frame["solar_mw"] = used[1]
    frame["backup_mw"] = used[2]
    frame["transmission_mw_km"] = grid @ links.length_km.to_numpy()
    frame["backup_mwh"] = fuel
    frame["demand_mwh"] = allocation["load"]
    country = frame.drop(columns="node_id").groupby("country").sum().reset_index()
    return frame, country


def optimal_split(intercept, slope, weights):
    """Minimize Var_w(a + s*b) on [0,1]; a flat objective selects 0.5."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    a = np.asarray(intercept, dtype=float)
    b = np.asarray(slope, dtype=float)
    a = a - w @ a
    b = b - w @ b
    variance = float(w @ (b * b))
    if variance <= 1e-24:
        return 0.5
    return float(np.clip(-(w @ (a * b)) / variance, 0, 1))


def price_assets(nodes, links, installed, allocation, annual_factor, prices):
    """Allocate the cost of each source asset before summing destinations."""
    n = len(nodes)
    used = np.zeros((n, len(COMPONENTS)))
    physical = np.zeros_like(used)
    offshore = nodes.is_offshore.to_numpy(dtype=bool)
    for index, technology, k, mask in (
        (0, "onwind", 0, ~offshore),
        (1, "offwind", 0, offshore),
        (2, "solar-utility", 1, np.ones(n, bool)),
        (3, "CCGT", 2, np.ones(n, bool)),
    ):
        if np.any(installed[k, mask] > 0):
            rate = prices.annual(technology, ("EUR/kW", "EUR/kW_e", "EUR/kWel"), 1000)
            physical[mask, index] = installed[k, mask] * rate
            used[:, index] = allocation["generation"][k, mask].sum(axis=0) * rate
    if np.any(allocation["produced"][2] > 0):
        fuel = prices.get("gas", "fuel", "EUR/MWh_th") / prices.get(
            "CCGT", "efficiency", "per unit"
        )
        physical[:, 4] = allocation["produced"][2] * annual_factor * fuel
        used[:, 4] = allocation["backup_energy"].sum(axis=0) * annual_factor * fuel
    rates = np.zeros((len(links), 3))
    for dc, technology, column in ((False, "HVAC overhead", 0), (True, "HVDC submarine", 1)):
        selected = links.is_hvdc.to_numpy(dtype=bool) == dc
        selected &= links.capacity_mw.to_numpy() > 0
        if selected.any():
            if dc and (technology, "investment") not in prices.table.index:
                technology = "HVDC underground"
            rates[selected, column] = links.length_km.to_numpy()[selected] * prices.annual(
                technology, "EUR/MW/km"
            )
    selected = links.is_hvdc.to_numpy(dtype=bool) & (links.capacity_mw.to_numpy() > 0)
    if selected.any():
        rates[selected, 2] = prices.annual("HVDC inverter pair", "EUR/MW")
    load_share = allocation["load"] / allocation["load"].sum()
    physical[:, 5:] = load_share[:, None] * (links.capacity_mw.to_numpy() @ rates)
    reverse = allocation["reverse"] @ rates
    forward = allocation["forward"] @ rates
    return physical, used, reverse, forward


def tables(nodes, costs, annual_load):
    node = pd.DataFrame(costs, columns=[f"{c}_eur_per_year" for c in COMPONENTS])
    node.insert(0, "node_id", nodes.node_id.to_numpy())
    node.insert(1, "country", nodes.country.to_numpy())
    node["load_mwh_per_year"] = annual_load
    node["total_eur_per_year"] = costs.sum(axis=1)
    country = (
        node.drop(columns="node_id")
        .groupby("country", sort=True)
        .sum(numeric_only=True)
        .reset_index()
    )
    for frame in (node, country):
        for name in (*COMPONENTS, "total"):
            frame[f"{name}_eur_per_mwh"] = np.divide(
                frame[f"{name}_eur_per_year"],
                frame.load_mwh_per_year,
                out=np.full(len(frame), np.nan),
                where=frame.load_mwh_per_year.to_numpy() > 0,
            )
    return node, country


def evaluate_costs(nodes, links, installed, allocation, hours, prices, split):
    factor = 8760 / hours
    load = allocation["load"] * factor
    physical, used, reverse, forward = price_assets(
        nodes, links, installed, allocation, factor, prices
    )
    if split == "minimize_wsd":
        importer = used.copy()
        importer[:, 5:] = reverse
        exporter = used.copy()
        exporter[:, 5:] = forward
        a = tables(nodes, importer, load)[1]
        b = tables(nodes, exporter, load)[1]
        active = a.load_mwh_per_year > 0
        split = optimal_split(
            a.loc[active, "total_eur_per_mwh"].to_numpy(),
            (b.total_eur_per_mwh - a.total_eur_per_mwh)[active].to_numpy(),
            a.loc[active, "load_mwh_per_year"].to_numpy(),
        )
    if not isinstance(split, (int, float)) or not 0 <= split <= 1:
        raise ValueError("split must be in [0,1] or 'minimize_wsd'")
    used[:, 5:] = (1 - split) * reverse + split * forward
    if not np.allclose(used.sum(axis=0), physical.sum(axis=0), rtol=1e-7, atol=1e-5):
        raise ArithmeticError("Used and installed component costs do not conserve")
    nt, ct = tables(nodes, used, load)
    ni, ci = tables(nodes, physical, load)
    active = ct.load_mwh_per_year > 0
    summary = {
        "split": float(split),
        "eu_lcoe_eur_per_mwh": float(used.sum() / load.sum()),
        "country_wsd_eur_per_mwh": weighted_dispersion(
            ct.loc[active, "total_eur_per_mwh"], ct.loc[active, "load_mwh_per_year"]
        ),
    }
    return nt, ct, ni, ci, summary
