"""Pinned public datasets used to prepare weather and usable land area."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import xarray as xr

CORINE_2018_URL = (
    "https://zenodo.org/records/13933243/files/U2018_CLC2018_V2020_20u1_3digits.tif?download=1"
)
CORINE_2018_LEGEND_URL = (
    "https://image.discomap.eea.europa.eu/arcgis/rest/services/"
    "Corine/CLC2018_WM/MapServer/legend?f=pjson"
)
GEBCO_2019_URL = (
    "https://tds.marine.rutgers.edu/thredds/dodsC/other/bathymetry/GEBCO_2019/GEBCO_2019.nc"
)
NATURAL_EARTH_10M_COUNTRIES_URL = (
    "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip"
)
NATURAL_EARTH_VERSION = "10m-admin-0"
CORINE_VERSION = "2018"
GEBCO_VERSION = "2019"


def sources(project, input_name, *, download, offshore, bounds=None):
    """Reuse original public downloads; acquire only missing physical sources."""
    import rasterio

    from . import storage as st

    project = Path(project)
    original = project / "derived_inputs" / input_name / "downloads"
    root = project / ".eurotracer" / "downloads"
    root.mkdir(parents=True, exist_ok=True)
    requested = (
        (
            "countries",
            original / "natural_earth/10m-admin-0/natural_earth_countries.zip",
            root / "natural_earth_countries.zip",
            NATURAL_EARTH_10M_COUNTRIES_URL,
        ),
        ("corine", original / "corine/2018/corine.tif", root / "corine_2018.tif", CORINE_2018_URL),
    )
    found = {}
    identities = {}
    for name, old, new, url in requested:
        path = old if old.is_file() else new
        if not path.is_file():
            if not download:
                raise FileNotFoundError(f"Missing {name}: {path}; enable download")
            _download_file(url, path)
        found[name] = path
        identities[name] = {"url": url, "sha256": st.file_hash(path)}
    gebco = None
    if offshore:
        if bounds is None:
            raise ValueError("GEBCO requires explicit bounds")
        for path in sorted(original.glob("gebco/2019/*/gebco.tif")):
            with rasterio.open(path) as raster:
                r = raster.bounds
                if (
                    r.left <= bounds[0]
                    and r.bottom <= bounds[1]
                    and r.right >= bounds[2]
                    and r.top >= bounds[3]
                ):
                    gebco = path
                    break
        if gebco is None:
            gebco = root / f"gebco_2019_{st.identity(bounds)}.tif"
            if not gebco.is_file():
                if not download:
                    raise FileNotFoundError(f"Missing offshore depth coverage: {gebco}")
                partial = gebco.with_suffix(".partial.tif")
                _download_gebco(bounds, partial)
                partial.replace(gebco)
        identities["gebco"] = {"url": GEBCO_2019_URL, "sha256": st.file_hash(gebco)}
    return found["countries"], found["corine"], gebco, identities


def _download_file(url: str, output: Path) -> None:
    """Download one URL into a temporary file before atomic replacement."""
    partial = output.with_name(f".{output.name}.download")
    partial.unlink(missing_ok=True)
    try:
        urllib.request.urlretrieve(url, partial)
        _require_nonempty(partial, output.name)
        partial.replace(output)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _require_nonempty(path: Path, label: str) -> None:
    """Check that a downloaded file exists and is not empty."""
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"{label} is missing or empty: {path}")


def _download_gebco(bounds: tuple[float, float, float, float], output: Path) -> None:
    """Download and crop the pinned GEBCO raster to the requested bounds."""
    try:
        import rioxarray  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("GEBCO preparation requires the data extra") from exc
    min_lon, min_lat, max_lon, max_lat = bounds
    dataset = xr.open_dataset(GEBCO_2019_URL, engine="netcdf4")
    try:
        latitude = dataset["lat"]
        latitude_slice = (
            slice(min_lat, max_lat)
            if float(latitude[0]) < float(latitude[-1])
            else slice(max_lat, min_lat)
        )
        elevation = dataset["elevation"].sel(lon=slice(min_lon, max_lon), lat=latitude_slice)
        elevation = elevation.sortby("lat", ascending=False).sortby("lon")
        elevation = elevation.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
        elevation = elevation.rio.write_crs("EPSG:4326")
        # Managed downloads use an atomic temporary path ending in
        # ``.download``. Rasterio therefore cannot infer the output format from
        # the filename even though the completed artifact is ``gebco.tif``.
        elevation.rio.to_raster(output, driver="GTiff")
    finally:
        dataset.close()
