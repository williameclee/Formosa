"""
Downloads digital elevation model data from the GMRT GridServer.

Last modified: 2026-08-10, En-Chi Lee (williameclee@gmail.com)
"""

import os
from pathlib import Path
import requests
import rasterio
from rasterio import Affine
from rasterio.io import MemoryFile
import numpy as np

from formosa.core import DATA_DIR
from formosa.dem.api.utils import number, _validate_latlon_limits, _dem_post_processing

from typing import Literal, TypeAlias, TypeVar, Iterable
import numpy.typing as npt

GmrtRes: TypeAlias = Literal["default", "med", "high", "max"]
GMRT_FMTS = ("netcdf", "coards", "esriascii", "geotiff")
GmrtFmt: TypeAlias = Literal["netcdf", "coards", "esriascii", "geotiff"]
gmrt_fmt_replacements: dict[str, GmrtFmt] = {"tiff": "geotiff", "netcdf4": "netcdf"}

GMRT_URL = "http://www.gmrt.org/services/GridServer?"
GMRT_LOCAL_DIR = DATA_DIR / "DEM" / "gmrt"
GMRT_RESS = ("default", "med", "high", "max")


def gmrt(
    latlim: tuple[number, number],
    lonlim: tuple[number, number],
    resolution: number | GmrtRes = "default",
    format: GmrtFmt = "geotiff",
    saveas: str | Path | None = "default path",
    forcenew: bool = False,
    base_url: str = GMRT_URL,
) -> tuple[
    npt.NDArray[np.floating | np.integer],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    Affine,
]:
    """
    Fetch DEM data from the GMRT server.
    For documentation of the API itself, see: https://www.gmrt.org/services/gridserverinfo.php#!/services/getGMRTGridURLs

    Parameters
    ----------
    latlim : tuple[number, number]
        Latitude limits (min, max) in degrees.
    lonlim : tuple[number, number]
        Longitude limits (min, max) in degrees.
    resolution : number | "default" | "med" | "high" | "max", optional
        Resolution of the DEM data. Can be a positive number or one of the predefined strings
        (default is "default").
    format : str, optional
        Format of the DEM data. Must be one of "netcdf", "coards",
        "esriascii", or "geotiff"
        (default is "geotiff").
    saveas : str | Path | None, optional
        Path to save the downloaded DEM file. If "default path", saves to åthe default path.
        If None, does not save the file
        (default is "default path").
    forcenew : bool, optional
        If True, forces a new download even if the file already exists
        (default is False).
    base_url : str, optional
        Base URL of the GMRT server
        (default is GMRT_URL).

    Returns
    -------
    Z : ndarray[floating | integer]
        2D array of elevation values.
    X : ndarray[floating]
        2D array of x-coordinates corresponding to Z.
    Y : ndarray[floating]
        2D array of y-coordinates corresponding to Z.
    transform : rasterio.Affine
        Affine transformation mapping pixel coordinates to spatial coordinates.

    Raises
    ------
    ValueError
        If input parameters are invalid or if no data is available for the specified bounds.
    ConnectionError
        If there is a failure in connecting to the GMRT server.
    FileNotFoundError
        If the requested data is not found on the GMRT server.
    """
    # Input validation
    latlim, lonlim = _validate_latlon_limits(latlim, lonlim)
    resolution = _validate_gmrt_resolution(resolution)
    format = _validate_gmrt_format(format)

    # Load data
    default_path = _gmrt_default_save_path(latlim, lonlim, resolution)

    # If the file exists and forcenew is False, load from file
    if not forcenew and os.path.exists(default_path):
        print(f"DEM file '{default_path}' already exists, skipping download")
        with rasterio.open(default_path) as src:
            Z = src.read(1)
            profile = src.profile
    else:
        Z, profile = _fetch_gmrt_data(
            latlim, lonlim, resolution, format, base_url=base_url
        )
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
    resolution: number | GmrtRes,
    accepted_resolutions: Iterable[str] = GMRT_RESS,
) -> number | GmrtRes:
    """
    Validate resolution input.
    """
    if isinstance(resolution, str):
        assert (
            resolution in accepted_resolutions
        ), f"Resolution as a string must be one of {accepted_resolutions} (got '{resolution}')"
    elif isinstance(resolution, (int, float)):
        assert (
            resolution > 0
        ), f"Resolution as a number must be positive (got {resolution})"
    return resolution


def _validate_gmrt_format(
    format: GmrtFmt,
    accepted_formats: Iterable[str] = GMRT_FMTS,
    format_replacements: dict[str, GmrtFmt] = gmrt_fmt_replacements,
) -> GmrtFmt:
    """
    Validate format input.
    """
    format = format.lower()  # type: ignore
    format = format_replacements.get(format, format)
    assert (
        format in accepted_formats
    ), f"Format must be one of {accepted_formats} (got '{format}')"
    return format


def _construct_gmrt_request(
    latlim: tuple[number, number],
    lonlim: tuple[number, number],
    resolution: number | str,
    format: str,
    layer: str = "topo",
) -> dict[str, str | number]:
    """
    Convert input parameters to GMRT request parameters.
    """
    params: dict[str, str | number] = {}
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
    latlim: tuple[number, number],
    lonlim: tuple[number, number],
    resolution: number | str,
    format: str,
    base_url: str = GMRT_URL,
) -> tuple[
    npt.NDArray[np.floating | np.integer],
    dict,
]:
    """
    Fetch DEM data from the GMRT server.
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
    resolution: number | str,
    dir: Path = GMRT_LOCAL_DIR,
) -> Path:
    """
    Generate the default local save path for GMRT DEM files.
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
