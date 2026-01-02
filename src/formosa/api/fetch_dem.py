from pathlib import Path
import requests
import affine
import rasterio
from rasterio import Affine
import rasterio.transform as rt
from rasterio.io import MemoryFile
from rasterio.profiles import Profile
import os

from formosa.core import DATA_DIR

import numpy as np

import warnings
from typing import Literal, TypeVar, Iterable
import numpy.typing as npt

number = TypeVar("number", int, float)

GMRT_URL = "http://www.gmrt.org/services/GridServer?"
GMRT_LOCAL_DIR = DATA_DIR / "DEM" / "gmrt"
gmrt_fmts = ("netcdf", "coards", "esriascii", "geotiff")
gmrt_fmt_replacements = {"tiff": "geotiff", "netcdf4": "netcdf"}
gmrt_ress = ("default", "med", "high", "max")


def gmrt(
    latlim: tuple[number, number],
    lonlim: tuple[number, number],
    resolution: number | Literal["default", "med", "high", "max"] = "default",
    format: str = "geotiff",
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
    resolution = _validate_resolution(resolution)
    format = _validate_format(format)

    # Load data
    default_path = _gmrt_default_save_path(latlim, lonlim, resolution)

    # If the file exists and forcenew is False, load from file
    if not forcenew and os.path.exists(default_path):
        print(f"[FORMOSA] File '{default_path}' already exists, skipping download")
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

    # Generate X, Y coordinate arrays
    transform = profile.get("transform", Affine.identity())
    ii, jj = np.meshgrid(
        np.arange(Z.shape[1]), np.arange(Z.shape[0])
    )  # x and y indices
    X, Y = rt.xy(transform, jj, ii)  # x and y coordinates
    X, Y = np.reshape(X, (-1,)).reshape(Z.shape), np.reshape(Y, (-1,)).reshape(Z.shape)

    return Z, X, Y, transform


def _validate_latlon_limits(
    latlim: tuple[number, number], lonlim: tuple[number, number]
) -> tuple[tuple[number, number], tuple[number, number]]:
    """
    Validate latitude and longitude limits.
    """
    # Latitude limits
    if latlim[0] > latlim[1]:
        latlim = (latlim[1], latlim[0])
        warnings.warn(
            f"Lower bound of latitude band ({latlim[0]}) was greater than upper bound ({latlim[1]}), swapping values."
        )
    elif latlim[0] == latlim[1]:
        raise ValueError(
            f"Lattidue band cannot have equal lower and upper bounds ({latlim[0]})."
        )
    # Longitude limits
    if lonlim[0] > lonlim[1]:
        lonlim = (lonlim[1], lonlim[0])
        warnings.warn(
            f"Lower bound of longitude band ({lonlim[0]}) was greater than upper bound ({lonlim[1]}), swapping values."
        )
    elif lonlim[0] == lonlim[1]:
        raise ValueError(
            f"Longitude band cannot have equal lower and upper bounds ({lonlim[0]})."
        )
    return latlim, lonlim


def _validate_resolution(
    resolution: number | Literal["default", "med", "high", "max"],
    accepted_resolutions: Iterable[str] = gmrt_ress,
) -> number | Literal["default", "med", "high", "max"]:
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


def _validate_format(
    format: str,
    accepted_formats: Iterable[str] = gmrt_fmts,
    format_replacements: dict[str, str] = gmrt_fmt_replacements,
) -> str:
    """
    Validate format input.
    """
    format = format.lower()
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
            "north": latlim[1],
            "south": latlim[0],
            "east": lonlim[1],
            "west": lonlim[0],
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
    Profile,
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


def dem_opentopo(
    latlim: tuple[float | int, float | int] = (20, 30),
    lonlim: tuple[float | int, float | int] = (120, 130),
    product: str = "SRTMGL1",
    format: str = "geotiff",
    saveas: str | None = "default path",
    api_key: str | None = None,
    forcenew: bool = False,
):
    ## Input validation
    if api_key is None:
        raise ValueError("[FORMOSA] API_KEY must be provided for OpenTopography")
    # Latitude/longitude limits
    if latlim[0] > latlim[1]:
        latlim = (latlim[1], latlim[0])
        print("[FORMOSA] Warning: LATLIM min is greater than max, swapping values")
    elif latlim[0] == latlim[1]:
        raise ValueError("[FORMOSA] LATLIM min and max cannot be the same value")
    if lonlim[0] > lonlim[1]:
        lonlim = (lonlim[1], lonlim[0])
        print("[FORMOSA] Warning: LONLIM min is greater than max, swapping values")
    elif lonlim[0] == lonlim[1]:
        raise ValueError("[FORMOSA] LONLIM min and max cannot be the same value")
    lon_offset = lonlim[0] // 360 * 360
    lonlim = (lonlim[0] - lon_offset, lonlim[1] - lon_offset)

    ## Main
    # Load data
    default_path = _opentopo_default_save_path(latlim, lonlim, product)
    if not forcenew and os.path.exists(default_path):
        print(f"[FORMOSA] File '{default_path}' already exists, skipping download")
        with rasterio.open(default_path) as src:
            data = src.read(1)
            profile = src.profile
    else:
        data, profile = _fetch_opentopo_data(
            latlim, lonlim, product, format, api_key, saveas
        )

    # Save data
    if saveas is not None:
        if saveas == "default path":
            saveas = default_path

        # Warn if the file already exists
        if os.path.exists(saveas):
            print(f"[FORMOSA] File '{saveas}' already exists and will be overwritten")

        with rasterio.open(saveas, "w", **profile) as dst:
            dst.write(data, 1)
        print(f"[FORMOSA] DEM saved to '{saveas}'")

    return data, profile


def _construct_opentopo_url(
    latlim: tuple[number, number],
    lonlim: tuple[number, number],
    product: str,
    format: str,
    api_key: str,
) -> str:
    match format.lower():
        case "tiff" | "geotiff":
            format = "GToff"

    base_url = "https://portal.opentopography.org/API/globaldem?"
    product_param = f"demtype={product}"
    aoi_param = f"north={latlim[1]}&south={latlim[0]}&east={lonlim[1]}&west={lonlim[0]}"
    format_param = f"outputFormat={format}"
    apikey_param = f"API_Key={api_key}"
    request_url = base_url + "&".join(
        [product_param, aoi_param, format_param, apikey_param]
    )
    print(request_url)
    return request_url


def _fetch_opentopo_data(
    latlim: tuple[number, number],
    lonlim: tuple[number, number],
    product: str,
    format: str,
    api_key: str,
    saveas: str | None = None,
):
    # Construct the URL
    request_url = _construct_opentopo_url(latlim, lonlim, product, format, api_key)

    # Retrieve the data
    response = requests.get(request_url)
    if response.status_code != 200:
        match response.status_code:
            case 204:
                raise ValueError(
                    f"[FORMOSA] No data in specified bounds on the OpenTopography server (204): {response.text}"
                )
            case 400:
                raise ValueError(
                    f"[FORMOSA] Bad request to OpenTopography server ({response.status_code}): {response.text}"
                )
            case _:
                raise ConnectionError(
                    f"[FORMOSA] Failed to fetch data from OpenTopography server ({response.status_code}): {response.text}"
                )

    with MemoryFile(response.content) as memfile:
        with memfile.open() as src:
            data = src.read(1)
            profile = src.profile

    if saveas is not None:
        if saveas == "default path":
            saveas = _opentopo_default_save_path(latlim, lonlim, product)

        # Warn if the file already exists
        if os.path.exists(saveas):
            print(f"[FORMOSA] File '{saveas}' already exists and will be overwritten")

        with rasterio.open(saveas, "w", **profile) as dst:
            dst.write(data, 1)
        print(f"[FORMOSA] DEM saved to '{saveas}'")
    return data, profile


def _opentopo_default_save_path(
    latlim: tuple[float | int, float | int],
    lonlim: tuple[float | int, float | int],
    product: str,
) -> str:
    product_param = "opentopo_" + product.lower()
    aoi_param = f"{latlim[0]}_{latlim[1]}_{lonlim[0]}_{lonlim[1]}"
    aoi_param = aoi_param.replace("-", "m").replace(".", "p")
    save_path = f"{product_param}-{aoi_param}.tiff"
    return save_path


def main():
    import matplotlib.pyplot as plt
    from formosa.cartography.colour import light_terrain

    # Example usage of gmrt
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
