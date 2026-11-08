"""Capped merit placement and independently optimized regional wind shares."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from . import storage as st
from .network import distances


def allocate(mean, limits, target, beta=1.0, *, capacity_target=False):
    """Water-filling of K ∝ mean(CF)**beta with exact MW or mean-MW target."""
    mean, limits = np.asarray(mean, dtype=float), np.asarray(limits, dtype=float)
    if (
        mean.shape != limits.shape
        or (mean < 0).any()
        or (limits < 0).any()
        or not np.isfinite(mean).all()
        or np.isnan(limits).any()
        or not np.isfinite(target)
        or target < 0
    ):
        raise ValueError("Invalid renewable allocation inputs")
    yield_ = np.ones_like(mean) if capacity_target else mean
    useful = mean > 0
    maximum = np.sum(limits[useful] * yield_[useful])
    tolerance = 1e-9 * max(1, target)
    if target > maximum + tolerance:
        raise ValueError(f"Renewable target {target:.6g} exceeds feasible {maximum:.6g}")
    result = np.zeros_like(mean)
    if target <= tolerance:
        return result
    weights = np.zeros_like(mean)
    weights[useful] = (mean[useful] / mean[useful].max()) ** beta
    active = useful & (limits > 0)
    for _ in range(len(mean) + 1):
        remaining = target - float(result @ yield_)
        if remaining <= tolerance:
            return result
        denominator = float((weights[active] * yield_[active]).sum())
        if denominator <= 0:
            raise ValueError("Positive generation target has no usable resource")
        proposal = remaining * weights / denominator
        room = np.maximum(limits - result, 0)
        result += np.where(active, np.minimum(proposal, room), 0)
        active &= result < limits - tolerance / max(1, float(mean.max(initial=0)))
    raise ArithmeticError("Capped placement failed to close its target")


def basis(data, progress=True):
    """Compute site means and durable nodal CFs once, independently of policy."""
    definition = {
        "weather": data.meta["weather"],
        "sites": data.meta["sites"],
        "load": data.meta["inputs"]["load_template.csv"],
        "nodes": data.meta["inputs"]["bus.csv"],
        "start": data.meta["start"],
        "stop": data.meta["stop"],
        "algorithm": 4,
    }
    path = data.project / ".eurotracer" / "basis" / st.identity(definition)
    with st.lock(path):
        if not (path / "complete.json").exists():
            if progress:
                print("Preparing nodal capacity factors and period means")
            z = st.group(path / "arrays.zarr", "a")
            out = [
                st.array(z, n, (len(data.time), len(data.nodes)), dtype="f8")
                for n in ("wind", "solar")
            ]
            node = data.sites.node.to_numpy()
            land = (node >= 0) & ~data.sites.is_sea.to_numpy(dtype=bool)
            counts = np.bincount(node[land], minlength=len(data.nodes))
            sums = np.zeros((2, len(data.sites)))
            load_sum = np.zeros(len(data.nodes))
            last_year = None
            for block in st.slices(len(data.time)):
                for technology, cf in enumerate(data.cf(block)):
                    sums[technology] += cf.sum(axis=0, dtype=float)
                    values = np.zeros((len(cf), len(data.nodes)))
                    np.add.at(values.T, node[land], cf[:, land].T)
                    # A coastal bus may own only sea-centered pixels. Preserve
                    # its load/network role without inventing an onshore resource.
                    out[technology][block] = np.divide(
                        values, counts, out=np.zeros_like(values), where=counts > 0
                    )
                load_sum += data.load(block).sum(axis=0)
                year = data.time[block.start].year
                if progress and year != last_year:
                    print(f"  nodal basis {year}: {block.stop:,}/{len(data.time):,} hours")
                    last_year = year
            st.save_arrays(
                path / "means.npz", site_cf=sums / len(data.time), load=load_sum / len(data.time)
            )
            st.write_json(path / "complete.json", definition)
    return path, st.group(path / "arrays.zarr"), st.load_arrays(path / "means.npz")


def medoids(lon, lat, weights, count):
    """Deterministic weighted medoids, distance rows evaluated in bounded groups."""
    if count < 1 or count > len(lon):
        raise ValueError("Hub count exceeds deployed sites")
    chosen = [int(np.argmax(weights))]
    while len(chosen) < count:
        distance = distances(lon[:, None], lat[:, None], lon[chosen], lat[chosen]).min(axis=1)
        distance[chosen] = -1
        chosen.append(int(np.argmax(distance * weights)))
    for _ in range(20):
        assignment = distances(lon[:, None], lat[:, None], lon[chosen], lat[chosen]).argmin(axis=1)
        updated = chosen.copy()
        for c in range(count):
            members = np.flatnonzero(assignment == c)
            if len(members):
                cost = np.empty(len(members))
                for block in st.slices(len(members), 128):
                    candidates = members[block]
                    cost[block] = (
                        distances(
                            lon[candidates, None], lat[candidates, None], lon[members], lat[members]
                        )
                        @ weights[members]
                    )
                updated[c] = int(members[np.argmin(cost)])
        if updated == chosen:
            break
        chosen = updated
    return np.asarray(chosen)


def offshore_layout(data, scenario, mean):
    nodes = data.nodes.copy()
    nodes["basin_id"] = ""
    mapping = np.full(len(data.sites), -1, dtype=int)
    installed = np.zeros(len(data.sites))
    policy = scenario.offshore
    if policy is None:
        return nodes, mapping, installed
    sites = data.sites
    sea = sites.is_sea.to_numpy(dtype=bool)
    eligible = (
        sea
        & sites.water_depth_m.between(policy.min_depth_m, policy.max_depth_m).to_numpy()
        & (sites.distance_to_land_m >= policy.land_setback_m).to_numpy()
    )
    caps = np.where(eligible, sites.cell_area_km2 * policy.density_mw_per_km2, 0)
    for basin in policy.basins:
        if basin.share == 0:
            continue
        members = np.flatnonzero(sites.basin_id.to_numpy() == basin.basin_id)
        if not len(members):
            raise ValueError(f"No prepared sites for offshore basin {basin.basin_id}")
        installed[members] = allocate(
            mean[members],
            caps[members],
            policy.capacity_mw * basin.share,
            policy.beta,
            capacity_target=True,
        )
        deployed = members[installed[members] > 0]
        x, y = sites.longitude.to_numpy(), sites.latitude.to_numpy()
        selected = medoids(
            x[deployed], y[deployed], installed[deployed] * mean[deployed], basin.hubs
        )
        hubs = deployed[selected]
        assignments = distances(x[deployed, None], y[deployed, None], x[hubs], y[hubs]).argmin(
            axis=1
        )
        land = np.flatnonzero(data.nodes.country.isin(basin.countries))
        if not len(land):
            raise ValueError(f"No permitted landfall countries for {basin.basin_id}")
        for h, site in enumerate(hubs):
            distance = distances(
                x[site],
                y[site],
                data.nodes.longitude.to_numpy()[land],
                data.nodes.latitude.to_numpy()[land],
            )
            proxy = land[np.lexsort((land, distance))[0]]
            number = len(nodes)
            nodes.loc[number] = [
                f"offshore-{basin.basin_id}-{h:03}",
                x[site],
                y[site],
                data.nodes.country.iloc[proxy],
                True,
                basin.basin_id,
            ]
            mapping[deployed[assignments == h]] = number
    return nodes, mapping, installed


def solve(data, scenario, progress=True):
    basis_path, node_cf, means = basis(data, progress)
    definition = {"basis": basis_path.name, "scenario": scenario.placement_values(), "algorithm": 3}
    path = data.project / ".eurotracer" / "placement" / st.identity(definition)
    with st.lock(path):
        if (path / "complete.json").exists():
            return path
        if progress:
            print(f"Solving {scenario.planning} renewable targets ({scenario.basis})")
        nodes, hub_map, offshore = offshore_layout(data, scenario, means["site_cf"][0])
        n = len(data.nodes)
        sites = data.sites
        land = (sites.node.to_numpy() >= 0) & ~sites.is_sea.to_numpy(dtype=bool)
        mapping = sites.node.to_numpy()
        site_caps = np.stack(
            (
                sites.land_wind_area_km2 * scenario.wind_density_mw_per_km2,
                sites.solar_area_km2 * scenario.solar_density_mw_per_km2,
            )
        )
        site_caps[:, ~land] = 0
        if not scenario.constrained:
            site_caps[:, land] = np.inf
        counts = np.bincount(mapping[land], minlength=n)
        node_mean = np.stack(
            [
                np.divide(
                    np.bincount(mapping[land], weights=m[land], minlength=n),
                    counts,
                    out=np.zeros(n),
                    where=counts > 0,
                )
                for m in means["site_cf"]
            ]
        )
        node_caps = np.stack(
            [np.bincount(mapping[land], weights=c[land], minlength=n) for c in site_caps]
        )
        countries = data.nodes.country.to_numpy()
        regions = sorted(set(countries)) if scenario.planning == "country" else ["EU"]
        group = (
            np.array([regions.index(c) for c in countries])
            if scenario.planning == "country"
            else np.zeros(n, dtype=int)
        )
        regional_load = np.bincount(group, weights=means["load"], minlength=len(regions))
        offmean = float(offshore @ means["site_cf"][0])
        targets = scenario.gamma * regional_load - offmean * regional_load / regional_load.sum()
        if (targets < -1e-7).any():
            raise ValueError("Offshore expected output exceeds gamma times demand")
        targets = np.maximum(targets, 0)
        installed_node = np.zeros((2, n))
        installed_site = np.zeros((2, len(sites)))
        alpha_values = {}
        diagnostics = []
        # Offshore output is computed once, never once per candidate alpha.
        offseries = np.zeros(len(data.time))
        if np.any(offshore):
            for block in st.slices(len(data.time)):
                offseries[block] = data.cf(block)[0] @ offshore
        for r, region in enumerate(regions):
            members = np.flatnonzero(group == r)
            pixels = np.flatnonzero(land & np.isin(mapping, members))
            direct = scenario.basis == "pixel" and scenario.pixel_scheme == "direct"
            mean = means["site_cf"][:, pixels] if direct else node_mean[:, members]
            caps = site_caps[:, pixels] if direct else node_caps[:, members]
            target = targets[r]
            maxgen = np.array(
                [np.sum(c[m > 0] * m[m > 0]) for c, m in zip(caps, mean, strict=True)]
            )
            low = max(0.0, 1 - maxgen[1] / target) if target else 0.0
            high = min(1.0, maxgen[0] / target) if target else 1.0
            if low > high + 1e-10:
                raise ValueError(f"{region}: wind and solar caps cannot meet gamma target")

            def capacities(alpha, mean=mean, caps=caps, target=target):
                result = np.stack(
                    [
                        allocate(mean[k], caps[k], target * f, beta)
                        for k, (f, beta) in enumerate(
                            ((alpha, scenario.wind_beta), (1 - alpha, scenario.solar_beta))
                        )
                    ]
                )
                return result

            # Read only this region's basis into a bounded disk-backed scratch array.
            scratch = path / f"candidate-{r}.npy"
            cf = np.lib.format.open_memmap(
                scratch, mode="w+", dtype="f8", shape=(len(data.time), len(mean[0]), 2)
            )
            load = np.empty(len(data.time))
            for block in st.slices(len(data.time)):
                raw = (
                    data.cf(block) if direct else (node_cf["wind"][block], node_cf["solar"][block])
                )
                for k in range(2):
                    cf[block, :, k] = raw[k][:, pixels if direct else members]
                load[block] = data.load(block)[:, members].sum(axis=1)

            def objective(alpha, cf=cf, r=r, load=load, region=region, capacities=capacities):
                k = capacities(alpha)
                energy = 0.0
                for block in st.slices(len(data.time), 720):
                    generation = cf[block, :, 0] @ k[0] + cf[block, :, 1] @ k[1]
                    generation += offseries[block] * regional_load[r] / regional_load.sum()
                    energy += np.maximum(load[block] - generation, 0).sum()
                diagnostics.append(
                    {"region": region, "alpha": float(alpha), "backup_mwh": float(energy)}
                )
                return float(energy)

            try:
                fixed = (
                    scenario.alpha.get(region)
                    if isinstance(scenario.alpha, dict)
                    else scenario.alpha
                )
                if isinstance(scenario.alpha, dict) and fixed is None:
                    raise ValueError(f"Missing fixed alpha for {region}")
                if fixed is not None:
                    if not low - 1e-10 <= fixed <= high + 1e-10:
                        raise ValueError(f"{region}: fixed alpha outside feasible [{low}, {high}]")
                    selected = float(fixed)
                    objective(selected)
                elif high - low < scenario.alpha_tolerance:
                    selected = (low + high) / 2
                    objective(selected)
                else:
                    # Endpoints and a coarse scan avoid relying on a unimodal cap response.
                    candidates = np.linspace(low, high, 21)
                    values = [objective(a) for a in candidates]
                    best = int(np.argmin(values))
                    selected = float(candidates[best])
                    for i in range(1, len(candidates) - 1):
                        if values[i] <= values[i - 1] and values[i] <= values[i + 1]:
                            fit = minimize_scalar(
                                objective,
                                bounds=(candidates[i - 1], candidates[i + 1]),
                                method="bounded",
                                options={"xatol": scenario.alpha_tolerance},
                            )
                            if not fit.success:
                                raise ArithmeticError(f"Alpha optimization failed for {region}")
                            if fit.fun < min(values):
                                selected = float(fit.x)
                                values.append(float(fit.fun))
                allocated = capacities(selected)
            finally:
                cf._mmap.close()
                scratch.unlink(missing_ok=True)
            alpha_values[region] = selected
            if direct:
                installed_site[:, pixels] = allocated
                for k in range(2):
                    installed_node[k] += np.bincount(
                        mapping[pixels], weights=allocated[k], minlength=n
                    )
            else:
                installed_node[:, members] = allocated
                if scenario.basis == "pixel":
                    for local, node in enumerate(members):
                        p = np.flatnonzero(land & (mapping == node))
                        for k, beta in enumerate(
                            (scenario.wind_beta_inner, scenario.solar_beta_inner)
                        ):
                            installed_site[k, p] = allocate(
                                means["site_cf"][k, p],
                                site_caps[k, p],
                                allocated[k, local] * node_mean[k, node],
                                beta,
                            )
                    # Pixel capacity differs from equivalent nodal capacity; targets do not.
                    for k in range(2):
                        installed_node[k, members] = np.bincount(
                            mapping[pixels], weights=installed_site[k, pixels], minlength=n
                        )[members]
            if progress:
                print(f"  {region}: alpha={selected:.4f}, target={target:,.1f} mean MW")
        installed = np.pad(installed_node, ((0, 0), (0, len(nodes) - n)))
        if np.any(offshore):
            installed[0] += np.bincount(
                hub_map[hub_map >= 0], weights=offshore[hub_map >= 0], minlength=len(nodes)
            )
        nodes.to_parquet(path / "nodes.parquet", index=False)
        st.save_arrays(
            path / "layout.npz",
            installed=installed,
            site_capacity=installed_site,
            offshore_capacity=offshore,
            hub_map=hub_map,
            mean_load=np.pad(means["load"], (0, len(nodes) - n)),
        )
        st.write_json(
            path / "complete.json",
            {
                **definition,
                "alpha": alpha_values,
                "diagnostics": diagnostics,
                "basis_path": str(basis_path),
            },
        )
    return path


def generation_block(data, scenario, layout, selection, node_cf):
    n = len(layout["mean_load"])
    result = np.zeros((selection.stop - selection.start, n, 2))
    raw = None
    if scenario.basis == "node":
        for k, name in enumerate(("wind", "solar")):
            result[:, : len(data.nodes), k] = (
                node_cf[name][selection] * layout["installed"][k, : len(data.nodes)]
            )
    else:
        raw = data.cf(selection)
        mapped = data.sites.node.to_numpy() >= 0
        for k in range(2):
            np.add.at(
                result[:, :, k].T,
                data.sites.node.to_numpy()[mapped],
                (raw[k][:, mapped] * layout["site_capacity"][k, mapped]).T,
            )
    if np.any(layout["offshore_capacity"]):
        wind = (raw or data.cf(selection))[0]
        active = layout["hub_map"] >= 0
        np.add.at(
            result[:, :, 0].T,
            layout["hub_map"][active],
            (wind[:, active] * layout["offshore_capacity"][active]).T,
        )
    return result
