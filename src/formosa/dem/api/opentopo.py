"""
Functions to download DEM data from the OpenTopography server.

Last modified: 2026-08-08, En-Chi Lee (williameclee@gmail.com)
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

from typing import Literal, TypeAlias, TypeVar
import numpy.typing as npt

OpenTopoProduct: TypeAlias = Literal[
    "SRTMGL3",
    "SRTMGL1",
    "SRTMGL1_E",
    "SRTM15Plus",
    "AW3D30",
    "AW3D30_E",
    "COP30",
    "COP90",
    "GEBCOIceTopo",
    "GEBCOSubIceTopo",
]

OPENTOPO_URL = "https://portal.opentopography.org/API/globaldem?"
OPENTOPO_LOCAL_DIR = DATA_DIR / "DEM" / "opentopo"


def opentopo(
    latlim: tuple[number, number],
    lonlim: tuple[number, number],
    api_key: str,
    product: OpenTopoProduct = "SRTMGL3",
    format: str = "geotiff",
    saveas: str | Path | None = "default path",
    forcenew: bool = False,
    base_url: str = OPENTOPO_URL,
) -> tuple[
    npt.NDArray[np.floating | np.integer],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    Affine,
]:
    """
    Fetch DEM data from the OpenTopography server.
    For documentation of the API itself, see: https://portal.opentopography.org/apidocs/#/Public/getGlobalDem

    Parameters
    ----------
    latlim : tuple[number, number]
        Latitude limits (min, max) in degrees.
    lonlim : tuple[number, number]
        Longitude limits (min, max) in degrees.
    api_key : str
        API key for accessing OpenTopography services.
    product : str, optional
        DEM product to fetch. Must be one of the supported products
        (default is "SRTMGL3").
    format : str, optional
        Format of the DEM data. Must be one of "netcdf", "coards",
        "esriascii", or "geotiff"
        (default is "geotiff").
    saveas : str | Path | None, optional
        Path to save the downloaded DEM file. If "default path", saves to the default path.
        If None, does not save the file
        (default is "default path").
    forcenew : bool, optional
        If True, forces a new download even if the file already exists
        (default is False).
    base_url : str, optional
        Base URL of the OpenTopography server
        (default is OPENTOPO_URL).

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
        If there is a failure in connecting to the OpenTopography server.
    FileNotFoundError
        If the requested data is not found on the OpenTopography server.
    """
    # Input validation
    if api_key is None:
        raise ValueError("API key must be provided for OpenTopography")
    latlim, lonlim = _validate_latlon_limits(latlim, lonlim)

    # Load data
    default_path = _opentopo_default_save_path(latlim, lonlim, product)
    if not forcenew and os.path.exists(default_path):
        print(f"DEM file '{default_path}' already exists, skipping download")
        with rasterio.open(default_path) as src:
            Z = src.read(1)
            profile = src.profile
    else:
        Z, profile = _fetch_opentopo_data(
            latlim, lonlim, product, format, api_key, opentopo_url=base_url
        )
        # Save data
        if saveas is not None:
            if saveas == "default path":
                saveas = default_path
            elif isinstance(saveas, str):
                saveas = Path(saveas)

            # Warn if the file already exists
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


def _construct_opentopo_url(
    latlim: tuple[number, number],
    lonlim: tuple[number, number],
    product: str,
    format: str,
    api_key: str,
) -> dict[str, str | number]:
    """
    Convert input parameters to OpenTopography request parameters.
    """
    match format.lower():
        case "tiff" | "geotiff":
            format = "GTiff"

    params: dict[str, str | number] = {}
    params.update(
        {
            "demtype": product,
            "north": latlim[1],
            "south": latlim[0],
            "east": lonlim[1],
            "west": lonlim[0],
            "outputFormat": format,
            "API_Key": api_key,
        }
    )
    return params


def _fetch_opentopo_data(
    latlim: tuple[number, number],
    lonlim: tuple[number, number],
    product: str,
    format: str,
    api_key: str,
    opentopo_url: str = OPENTOPO_URL,
):
    # Construct the URL
    params = _construct_opentopo_url(latlim, lonlim, product, format, api_key)

    # Retrieve the data
    response = requests.get(opentopo_url, params=params)
    if response.status_code != 200:
        match response.status_code:
            case 204:
                raise ValueError(
                    f"No data in specified bounds on the OpenTopography server (204): {response.text}"
                )
            case 400:
                raise ValueError(
                    f"Bad request to OpenTopography server (400): {response.text}"
                )
            case _:
                raise ConnectionError(
                    f"Failed to fetch data from OpenTopography server ({response.status_code}): {response.text}"
                )

    with MemoryFile(response.content) as memfile:
        with memfile.open() as src:
            data = src.read(1)
            profile = src.profile
    return data, profile


def _opentopo_default_save_path(
    latlim: tuple[float | int, float | int],
    lonlim: tuple[float | int, float | int],
    product: str,
    dir: Path = OPENTOPO_LOCAL_DIR,
) -> Path:
    """
    Generate the default local save path for OpenTopography DEM files.
    """
    product_param = "opentopo_" + product.lower()
    aoi_param = f"{latlim[0]}_{latlim[1]}_{lonlim[0]}_{lonlim[1]}"
    aoi_param = aoi_param.replace("-", "m").replace(".", "p")
    save_file = f"{product_param}-{aoi_param}.tiff"
    return dir / save_file
