import requests
import affine
import rasterio
import rasterio.transform as rt
from rasterio.io import MemoryFile
import os

import numpy as np

from typing import Literal, TypeVar
import numpy.typing as npt

accepted_formats = ("netcdf", "coards", "esriascii", "geotiff")
accepted_resolutions = ("default", "med", "high", "max")

number = TypeVar("number", int, float)


def dem_gmrt(
    latlim: tuple[number, number] = (20, 30),
    lonlim: tuple[number, number] = (120, 130),
    resolution: number | Literal["default", "med", "high", "max"] = "default",
    format: str = "geotiff",
    saveas: str | None = "default path",
    forcenew: bool = False,
) -> tuple[
    npt.NDArray[np.float32],
    dict,
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
]:
    ## Input validation
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

    # Resolution
    if isinstance(resolution, str):
        assert (
            resolution in accepted_resolutions
        ), f"[FORMOSA] RESOLUTION as a string must be one of {accepted_resolutions}, got '{resolution}' instead"
    elif isinstance(resolution, (int, float)):
        assert (
            resolution > 0
        ), f"[FORMOSA] RESOLUTION as a number must be positive, got {resolution} instead"

    # Format
    format = format.lower()
    replacement_formats = {"tiff": "geotiff", "netcdf4": "netcdf"}
    format = replacement_formats.get(format, format)
    assert (
        format in accepted_formats
    ), f"[FORMOSA] FORMAT must be one of {accepted_formats}, got '{format}' instead"

    ## Main
    # Load data
    default_path = _gmrt_default_save_path(latlim, lonlim, resolution)
    if not forcenew and os.path.exists(default_path):
        print(f"[FORMOSA] File '{default_path}' already exists, skipping download")
        with rasterio.open(default_path) as src:
            data = src.read(1)
            profile = src.profile
    else:
        data, profile = _fetch_gmrt_data(latlim, lonlim, resolution, format, saveas)

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

    transform = profile.get("transform", affine.Affine.identity())
    ii, jj = np.meshgrid(
        np.arange(data.shape[1]), np.arange(data.shape[0])
    )  # x and y indices
    xx, yy = rt.xy(transform, jj, ii)  # x and y coordinates
    xx, yy = np.reshape(xx, (-1,)).reshape(data.shape), np.reshape(yy, (-1,)).reshape(
        data.shape
    )

    return data, profile, (xx, yy)


def _construct_gmrt_url(
    latlim: tuple[number, number],
    lonlim: tuple[number, number],
    resolution: number | str,
    format: str,
    layer: str = "topo",
) -> str:
    base_url = "http://www.gmrt.org/services/GridServer?"
    aoi_param = f"north={latlim[1]}&south={latlim[0]}&east={lonlim[1]}&west={lonlim[0]}"
    format_param = f"format={format}"
    resolution_param = f"resolution={resolution}"
    layer_param = f"layer={layer}"
    mformat_param = "mformat=xml"
    request_url = base_url + "&".join(
        [aoi_param, format_param, resolution_param, layer_param, mformat_param]
    )
    return request_url


def _fetch_gmrt_data(
    latlim: tuple[number, number],
    lonlim: tuple[number, number],
    resolution: number | str,
    format: str,
    saveas: str | None = None,
):
    # Construct the URL
    request_url = _construct_gmrt_url(latlim, lonlim, resolution, format)

    # Retrieve the data
    response = requests.get(request_url)
    if response.status_code != 200:
        match response.status_code:
            case 404:
                raise FileNotFoundError(
                    f"[FORMOSA] An error has occured on the GMRT server ({response.status_code}): {response.text}"
                )
            case 204:
                raise ValueError(
                    f"[FORMOSA] No data in specified bounds on the GMRT server ({response.status_code}): {response.text}"
                )
            case 413:
                raise ValueError(
                    f"[FORMOSA] The requested area is too large ({response.status_code}): {response.text}"
                )
            case _:
                raise ConnectionError(
                    f"[FORMOSA] Failed to fetch data from GMRT server ({response.status_code}): {response.text}"
                )

    with MemoryFile(response.content) as memfile:
        with memfile.open() as src:
            data = src.read(1)
            profile = src.profile

    if saveas is not None:
        if saveas == "default path":
            saveas = _gmrt_default_save_path(latlim, lonlim, resolution)

        # Warn if the file already exists
        if os.path.exists(saveas):
            print(f"[FORMOSA] File '{saveas}' already exists and will be overwritten")

        with rasterio.open(saveas, "w", **profile) as dst:
            dst.write(data, 1)
        print(f"[FORMOSA] DEM saved to '{saveas}'")
    return data, profile


def _gmrt_default_save_path(
    latlim: tuple[float | int, float | int],
    lonlim: tuple[float | int, float | int],
    resolution: number | str,
) -> str:
    product_param = "gmrt"
    aoi_param = f"{latlim[0]}_{latlim[1]}_{lonlim[0]}_{lonlim[1]}"
    aoi_param = aoi_param.replace("-", "m").replace(".", "p")
    resolution_param = f"{resolution}".replace(".", "p")
    save_path = f"{product_param}-{aoi_param}-{resolution_param}.tiff"
    return save_path


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
