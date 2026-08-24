"""
Downloads digital elevation model data from the GMRT GridServer.

Last modified: 2026-08-23, En-Chi Lee (williameclee@gmail.com)
"""

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import rasterio
import requests
from numpy.typing import NDArray
from rasterio import Affine
from rasterio.io import MemoryFile

from formosa.core import DATA_DIR
from formosa.dem.api.utils import _dem_post_processing, _validate_latlon_limits
from formosa.utils import Real

GmrtRes: TypeAlias = Literal["default", "med", "high", "max"]
GMRT_FMTS = ("netcdf", "coards", "esriascii", "geotiff")
GmrtFmt: TypeAlias = Literal["netcdf", "coards", "esriascii", "geotiff"]
gmrt_fmt_replacements: dict[str, GmrtFmt] = {"tiff": "geotiff", "netcdf4": "netcdf"}

GMRT_URL = "http://www.gmrt.org/services/GridServer?"
GMRT_LOCAL_DIR = DATA_DIR / "DEM" / "gmrt"
GMRT_RESS = ("default", "med", "high", "max")


def gmrt(
    latlim: tuple[Real, Real],
    lonlim: tuple[Real, Real],
    res: Real | GmrtRes = "default",
    fmt: GmrtFmt = "geotiff",
    saveas: str | Path | None = "default path",
    forcenew: bool = False,
    base_url: str = GMRT_URL,
) -> tuple[
    NDArray[np.floating | np.integer],
    NDArray[np.floating],
    NDArray[np.floating],
    Affine,
]:
    """
    Fetches DEM data from the GMRT server.

    Parameters
    ----------
    latlim : tuple[number, number]
        Latitude limits (min, max) in degrees.
    lonlim : tuple[number, number]
        Longitude limits (min, max) in degrees.
    res : number | {"default", "med", "high", "max"}, optional
        Resolution of the DEM data. Can be a positive number or one
        of the predefined strings.
        - Default resolution is `"default"`.
    fmt : {"netcdf", "coards", "esriascii", "geotiff"}, optional
        Format of the DEM data.
        - Default format is `"geotiff"`.
    saveas : str | Path | None, optional
        Path to save the downloaded DEM file.
        If `"default path"`, saves to the default path.
        If `None`, does not save the file.
        - Default path is `"default path"`.
    forcenew : bool, optional
        Whether to force a new download even if the file exists.
        - Default option is `False`.
    base_url : str, optional
        Base URL of the GMRT server.
        - Default URL is `GMRT_URL`.

    Returns
    -------
    dem : NDArray[number]
        2D array of elevation values.
        - Shape: `(nrows, ncols)`.
    x : NDArray[float]
        x-coordinates corresponding to `dem`.
        - Shape: `(nrows, ncols)`, same as `dem`.
    y : NDArray[float]
        y-coordinates corresponding to `dem`.
        - Shape: `(nrows, ncols)`, same as `dem`.
    transform : rasterio.Affine
        Affine transformation mapping pixel coordinates to spatial
        coordinates.

    Raises
    ------
    ValueError
        If input parameters are invalid or if no data is available
        for the specified bounds.
    ConnectionError
        If there is a failure in connecting to the GMRT server.
    FileNotFoundError
        If the requested data is not found on the GMRT server.

    Notes
    -----
    For documentation of the API itself, see:
    https://www.gmrt.org/services/gridserverinfo.php#!/services/getGMRTGridURLs
    """
    # Input validation
    latlim, lonlim = _validate_latlon_limits(latlim, lonlim)
    res = _validate_gmrt_resolution(res)
    fmt = _validate_gmrt_format(fmt)

    # Load data
    default_path = _gmrt_default_save_path(latlim, lonlim, res)

    # If the file exists and forcenew is False, load from file
    if not forcenew and os.path.exists(default_path):
        print(f"DEM file '{default_path}' already exists, skipping download")
        with rasterio.open(default_path) as src:
            Z = src.read(1)
            profile = src.profile
    else:
        Z, profile = _fetch_gmrt_data(latlim, lonlim, res, fmt, base_url=base_url)
        # Save data
        if saveas is not None:
            if saveas == "default path":
                saveas = default_path
            elif isinstance(saveas, str):
                saveas = Path(saveas)
            if os.path.exists(saveas):
                print(f"DEM file '{saveas}' already exists and will be overwritten.")
            elif not saveas.parent.exists():
                saveas.parent.mkdir(parents=True, exist_ok=True)

            with rasterio.open(saveas, "w", **profile) as dst:
                dst.write(Z, 1)
            print(f"DEM saved to '{saveas}'")

    # Post-processing
    Z, X, Y, transform = _dem_post_processing(Z, profile)

    return Z, X, Y, transform


def _validate_gmrt_resolution(
    res: Real | GmrtRes,
    accepted_ress: Iterable[str] = GMRT_RESS,
) -> Real | GmrtRes:
    """
    Validate resolution input.
    """
    if isinstance(res, str):
        assert res in accepted_ress, (
            f"Resolution as a string must be one of {accepted_ress} (got '{res}')"
        )
    elif isinstance(res, (int, float)):
        assert res > 0, f"Resolution as a number must be positive (got {res})"
    return res


def _validate_gmrt_format(
    fmt: GmrtFmt,
    accepted_formats: Iterable[str] = GMRT_FMTS,
    format_replacements: dict[str, GmrtFmt] = gmrt_fmt_replacements,
) -> GmrtFmt:
    """
    Validates format input.
    """
    fmt = fmt.lower()  # type: ignore
    fmt = format_replacements.get(fmt, fmt)
    assert fmt in accepted_formats, (
        f"Format must be one of {accepted_formats} (got '{fmt}')"
    )
    return fmt


def _construct_gmrt_request(
    latlim: tuple[Real, Real],
    lonlim: tuple[Real, Real],
    resolution: Real | str,
    format: str,
    layer: str = "topo",
) -> dict[str, str | Real]:
    """
    Converts input parameters to GMRT request parameters.
    """
    params: dict[str, str | Real] = {}
    params.update(
        {
            "maxlatitude": latlim[1],
            "minlatitude": latlim[0],
            "maxlongitude": lonlim[1],
            "minlongitude": lonlim[0],
            "format": format,
            "resolution": resolution,
            "layer": layer,
            "mformat": "xml",
        }
    )
    return params


def _fetch_gmrt_data(
    latlim: tuple[Real, Real],
    lonlim: tuple[Real, Real],
    resolution: Real | str,
    format: str,
    base_url: str = GMRT_URL,
) -> tuple[
    NDArray[np.floating | np.integer],
    dict,
]:
    """
    Fetches DEM data from the GMRT server.
    """
    # Construct the URL
    params = _construct_gmrt_request(latlim, lonlim, resolution, format)

    # Retrieve the data
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        match response.status_code:
            case 404:
                raise FileNotFoundError(
                    f"An error has occured on the GMRT server ({response.status_code}): {response.text}"
                )
            case 204:
                raise ValueError(
                    f"No data in specified bounds on the GMRT server ({response.status_code}): {response.text}"
                )
            case 413:
                raise ValueError(
                    f"The requested area is too large ({response.status_code}): {response.text}"
                )
            case _:
                raise ConnectionError(
                    f"Failed to fetch data from GMRT server ({response.status_code}): {response.text}"
                )

    with MemoryFile(response.content) as memfile:
        with memfile.open() as src:
            data = src.read(1)
            profile = src.profile
    return data, profile


def _gmrt_default_save_path(
    latlim: tuple[float | int, float | int],
    lonlim: tuple[float | int, float | int],
    resolution: Real | str,
    dir: Path = GMRT_LOCAL_DIR,
) -> Path:
    """
    Generates the default local save path for GMRT DEM files.
    """
    product_param = "gmrt"
    aoi_param = f"{latlim[0]}_{latlim[1]}_{lonlim[0]}_{lonlim[1]}"
    aoi_param = aoi_param.replace("-", "m").replace(".", "p")
    resolution_param = f"{resolution}".replace(".", "p")
    if resolution_param.startswith("0p"):
        resolution_param = resolution_param[1:]
    save_file = f"{product_param}-{aoi_param}-{resolution_param}.tiff"
    return dir / save_file


def main():
    import matplotlib.pyplot as plt

    from formosa.graphics.colour import light_terrain

    # Example usage of GMRT
    z, x, y, _ = gmrt(
        latlim=(22, 24),
        lonlim=(121, 123),
        saveas=None,
    )

    plt.pcolormesh(x, y, z, shading="auto", cmap=light_terrain())
    plt.colorbar(label="Elevation [m]")
    plt.xlabel("Longitude [°E]")
    plt.ylabel("Latitude [°N]")
    plt.title("GMRT DEM")
    plt.show()


if __name__ == "__main__":
    main()
