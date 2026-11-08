"""Scientific assumptions. Presets are values, not separate model implementations."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Weather:
    turbine: str = "Vestas_V164_7MW_offshore"
    hub_height_m: float = 130.0
    smooth: bool = True
    interpolation: str = "power"
    panel: str = "CSi"
    domain_buffer_degrees: float = 0.5
    excluder_resolution_m: float = 300.0
    wind_excluded_codes: tuple[int, ...] = (
        111,
        112,
        123,
        124,
        411,
        412,
        421,
        422,
        423,
        511,
        512,
        521,
        522,
        523,
    )
    solar_excluded_codes: tuple[int, ...] = (
        111,
        112,
        311,
        312,
        313,
        411,
        412,
        421,
        422,
        423,
        511,
        512,
        521,
        522,
        523,
    )
    continuous_urban_setback_m: float = 2000.0
    discontinuous_urban_setback_m: float = 1000.0

    def __post_init__(self):
        for name in ("hub_height_m", "excluder_resolution_m"):
            positive(getattr(self, name), name)
        for name in (
            "domain_buffer_degrees",
            "continuous_urban_setback_m",
            "discontinuous_urban_setback_m",
        ):
            positive(getattr(self, name), name, zero=True)
        if self.interpolation not in ("power", "logarithmic"):
            raise ValueError("interpolation must be power or logarithmic")


@dataclass(frozen=True)
class Basin:
    basin_id: str
    share: float
    hubs: int
    landfalls: int
    countries: tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(self, "countries", tuple(self.countries))
        if any(type(v) is not int or v < 0 for v in (self.hubs, self.landfalls)):
            raise ValueError("hubs and landfalls must be nonnegative integers")
        if not self.basin_id or not 0 <= self.share <= 1:
            raise ValueError("basin requires an ID and share in [0, 1]")
        if self.share > 0 and (self.hubs < 1 or self.landfalls < 1 or not self.countries):
            raise ValueError("positive basin share requires hubs, landfalls and countries")
        if self.share == 0 and (self.hubs or self.landfalls or self.countries):
            raise ValueError("zero-share basin must have no hubs or landfalls")


@dataclass(frozen=True)
class Offshore:
    capacity_mw: float
    basins: tuple[Basin, ...]
    density_mw_per_km2: float = 8.0
    min_depth_m: float = 5.0
    max_depth_m: float = 60.0
    land_setback_m: float = 1000.0
    beta: float = 1.0

    def __post_init__(self):
        object.__setattr__(
            self, "basins", tuple(Basin(**b) if isinstance(b, dict) else b for b in self.basins)
        )
        for name in ("capacity_mw", "density_mw_per_km2"):
            positive(getattr(self, name), name)
        for name in ("min_depth_m", "max_depth_m", "land_setback_m", "beta"):
            positive(getattr(self, name), name, zero=True)
        if self.max_depth_m < self.min_depth_m:
            raise ValueError("offshore depth bounds are reversed")
        if (
            not self.basins
            or len({b.basin_id for b in self.basins}) != len(self.basins)
            or not math.isclose(sum(b.share for b in self.basins), 1, abs_tol=1e-9)
        ):
            raise ValueError("offshore basin IDs must be unique and shares must sum to one")


@dataclass(frozen=True)
class Scenario:
    """Inspect or replace assumptions with dataclasses.replace.

    C applies eligible-area caps, U removes them. LLnoT is domestic-only;
    LL balances countries, LG balances Europe, GG also plans Europe jointly.
    alpha=None chooses each planning region's backup-energy minimum.
    """

    preset: str = "GG-C"
    gamma: float = 1.0
    alpha: float | dict[str, float] | None = None
    wind_density_mw_per_km2: float = 7.0
    solar_density_mw_per_km2: float = 25.0
    wind_beta: float = 1.0
    solar_beta: float = 1.0
    basis: str = "node"
    pixel_scheme: str = "bus_avg"
    wind_beta_inner: float = 1.0
    solar_beta_inner: float = 1.0
    capacity_quantile: float = 0.99
    hvdc_fraction: float = 0.05
    length_exponent: float | str = 0.0
    length_exponent_bounds: tuple[float, float] = (0.0, 2.0)
    alpha_tolerance: float = 1e-4
    offshore: Offshore | None = None

    def __post_init__(self):
        if self.preset not in {f"{p}-{s}" for p in ("GG", "LG", "LL", "LLnoT") for s in ("C", "U")}:
            raise ValueError("preset must be GG/LG/LL/LLnoT followed by -C or -U")
        if isinstance(self.offshore, dict):
            object.__setattr__(self, "offshore", Offshore(**self.offshore))
        if self.basis not in ("node", "pixel") or self.pixel_scheme not in ("bus_avg", "direct"):
            raise ValueError("basis must be node/pixel; pixel_scheme must be bus_avg/direct")
        for name in (
            "gamma",
            "wind_density_mw_per_km2",
            "solar_density_mw_per_km2",
            "alpha_tolerance",
        ):
            positive(getattr(self, name), name)
        for name in ("wind_beta", "solar_beta", "wind_beta_inner", "solar_beta_inner"):
            positive(getattr(self, name), name, zero=True)
        for name in ("capacity_quantile", "hvdc_fraction"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.alpha is not None:
            values = self.alpha.values() if isinstance(self.alpha, dict) else (self.alpha,)
            if any(not 0 <= a <= 1 for a in values):
                raise ValueError("alpha must be in [0, 1]")
        if self.length_exponent != "minimize_volume":
            if not isinstance(self.length_exponent, (int, float)):
                raise ValueError("length_exponent must be numeric or 'minimize_volume'")
            positive(self.length_exponent, "length_exponent", zero=True)
        bounds = self.length_exponent_bounds
        if not (len(bounds) == 2 and 0 <= bounds[0] < bounds[1] and math.isfinite(bounds[1])):
            raise ValueError("invalid length_exponent_bounds")

    @property
    def planning(self):
        return "eu" if self.preset.startswith("GG") else "country"

    @property
    def balancing(self):
        return "country" if self.preset.startswith("LL") else "eu"

    @property
    def domestic_only(self):
        return self.preset.startswith("LLnoT")

    @property
    def constrained(self):
        return self.preset.endswith("-C")

    def describe(self):
        return {
            **asdict(self),
            "planning": self.planning,
            "balancing": self.balancing,
            "domestic_only": self.domestic_only,
            "constrained": self.constrained,
        }

    def placement_values(self):
        values = asdict(self)
        for name in (
            "preset",
            "capacity_quantile",
            "hvdc_fraction",
            "length_exponent",
            "length_exponent_bounds",
        ):
            values.pop(name)
        return {**values, "planning": self.planning, "constrained": self.constrained}


@dataclass(frozen=True)
class CostAssumptions:
    rate: float = 0.04
    overrides: dict[str, dict[str, float]] = field(default_factory=dict)

    def __post_init__(self):
        positive(self.rate, "rate", zero=True)
        for technology, params in self.overrides.items():
            for parameter, value in params.items():
                positive(value, f"{technology}/{parameter}", zero=True)


def positive(value, name, *, zero=False):
    if not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
