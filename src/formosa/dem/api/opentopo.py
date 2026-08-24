"""
Downloads digital elevation model data from OpenTopography.

Last modified: 2026-08-23, En-Chi Lee (williameclee@gmail.com)
"""

import os
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
    latlim: tuple[Real, Real],
    lonlim: tuple[Real, Real],
    api_key: str,
    product: OpenTopoProduct = "SRTMGL3",
    fmt: str = "geotiff",
    saveas: str | Path | None = "default path",
    forcenew: bool = False,
    base_url: str = OPENTOPO_URL,
) -> tuple[
    NDArray[np.floating | np.integer],
    NDArray[np.floating],
    NDArray[np.floating],
    Affine,
]:
    """
    Fetches DEM data from the OpenTopography server.

    Parameters
    ----------
    latlim : tuple[number, number]
        Latitude limits (min, max) in degrees.
    lonlim : tuple[number, number]
        Longitude limits (min, max) in degrees.
    api_key : str
        API key for accessing OpenTopography services.
    product : str, optional
        DEM product to fetch. Must be one of the supported products.
        - Default product is `"SRTMGL3"`.
    fmt : str, optional
        Format of the DEM data. Must be one of "netcdf", "coards",
        "esriascii", or "geotiff".
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
        Base URL of the OpenTopography server.
        - Default URL is `OPENTOPO_URL`.

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
        If there is a failure in connecting to the OpenTopography
        server.
    FileNotFoundError
        If the requested data is not found on the OpenTopography
        server.

    Notes
    -----
    For documentation of the API itself, see:
    https://portal.opentopography.org/apidocs/#/Public/getGlobalDem
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
            latlim, lonlim, product, fmt, api_key, opentopo_url=base_url
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
    latlim: tuple[Real, Real],
    lonlim: tuple[Real, Real],
    product: str,
    fmt: str,
    api_key: str,
) -> dict[str, str | Real]:
    """
    Converts input parameters to OpenTopography request parameters.
    """
    match fmt.lower():
        case "tiff" | "geotiff":
            fmt = "GTiff"

    params: dict[str, str | Real] = {}
    params.update(
        {
            "demtype": product,
            "north": latlim[1],
            "south": latlim[0],
            "east": lonlim[1],
            "west": lonlim[0],
            "outputFormat": fmt,
            "API_Key": api_key,
        }
    )
    return params


def _fetch_opentopo_data(
    latlim: tuple[Real, Real],
    lonlim: tuple[Real, Real],
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
    Generates the default local save path for OpenTopography DEM files.
    """
    product_param = "opentopo_" + product.lower()
    aoi_param = f"{latlim[0]}_{latlim[1]}_{lonlim[0]}_{lonlim[1]}"
    aoi_param = aoi_param.replace("-", "m").replace(".", "p")
    save_file = f"{product_param}-{aoi_param}.tiff"
    return dir / save_file
