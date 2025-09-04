import requests
import rasterio
from rasterio.io import MemoryFile
import os

accepted_formats = ("netcdf", "coards", "esriascii", "geotiff")
accepted_resolutions = ("default", "med", "high", "max")


def dem_gmrt(
    latlim: tuple[float | int, float | int] = (20, 30),
    lonlim: tuple[float | int, float | int] = (120, 130),
    resolution: int | float | str = "default",
    format: str = "geotiff",
    saveas: str | None = "default path",
    forcenew: bool = False,
):
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
        resolution = resolution.lower()
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
    default_path = _default_save_path(latlim, lonlim, resolution)
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

    return data, profile


def _construct_gmrt_url(
    latlim: tuple[int | float, int | float],
    lonlim: tuple[int | float, int | float],
    resolution: int | float | str,
    format: str,
    layer: str = "topo",
) -> str:
    base_url = "http://www.gmrt.org/services/GridServer?"
    aoi_param = f"north={latlim[1]}&south={latlim[0]}&east={lonlim[1]}&west={lonlim[0]}"
    format_param = f"&format={format}"
    resolution_param = f"&resolution={resolution}"
    layer_param = f"&layer={layer}"
    mformat_param = "&mformat=xml"
    request_url = (
        base_url
        + aoi_param
        + format_param
        + resolution_param
        + layer_param
        + mformat_param
    )
    return request_url


def _fetch_gmrt_data(latlim, lonlim, resolution, format, saveas):
    # Construct the URL
    request_url = _construct_gmrt_url(latlim, lonlim, resolution, format)

    # Retrieve the data
    response = requests.get(request_url)
    if response.status_code != 200:
        match response.status_code:
            case 404:
                raise FileNotFoundError(
                    f"[FORMOSA] GMRT resource not found at {request_url}"
                )
            case _:
                raise ConnectionError(
                    f"[FORMOSA] Failed to fetch data from GMRT server, status code {response.status_code}"
                )

    with MemoryFile(response.content) as memfile:
        with memfile.open() as src:
            data = src.read(1)
            profile = src.profile

    if saveas is not None:
        if saveas == "default path":
            saveas = _default_save_path(latlim, lonlim, resolution)

        # Warn if the file already exists
        if os.path.exists(saveas):
            print(f"[FORMOSA] File '{saveas}' already exists and will be overwritten")

        with rasterio.open(saveas, "w", **profile) as dst:
            dst.write(data, 1)
        print(f"[FORMOSA] DEM saved to '{saveas}'")
    return data, profile


def _default_save_path(
    latlim: tuple[float | int, float | int],
    lonlim: tuple[float | int, float | int],
    resolution: int | float | str,
) -> str:
    product_param = "gmrt"
    aoi_param = f"{latlim[0]}_{latlim[1]}_{lonlim[0]}_{lonlim[1]}"
    aoi_param = aoi_param.replace("-", "m").replace(".", "p")
    resolution_param = f"{resolution}".replace(".", "p")
    save_path = f"{product_param}-{aoi_param}-{resolution_param}.tiff"
    return save_path


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    data, profile = dem_gmrt(
        (20, 25),
        (120, 125),
        resolution="default",
        format="geotiff",
        saveas="default path",
        forcenew=False,
    )

    print(type(profile))
    print(profile)

    plt.imshow(data, cmap="terrain")
    plt.colorbar()
    plt.show()
