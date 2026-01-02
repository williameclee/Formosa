import warnings

from typing import TypeVar

number = TypeVar("number", int, float)


def _validate_latlon_limits(
    latlim: tuple[number, number], lonlim: tuple[number, number], format: bool = True
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

    if format:
        lon_offset = lonlim[0] // 360 * 360
        lonlim = (lonlim[0] - lon_offset, lonlim[1] - lon_offset)
    return latlim, lonlim


import numpy as np
import rasterio.transform as rt
from rasterio.transform import Affine
import numpy.typing as npt


def _dem_post_processing(
    Z: npt.NDArray[np.floating | np.integer], profile: dict
) -> tuple[
    npt.NDArray[np.floating | np.integer],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    Affine,
]:
    """
    Common DEM post-processing steps.
    """
    # Replace no data values with NaN
    no_data_value = profile.get("nodata", None)
    if no_data_value is not None:
        Z = np.where(Z == no_data_value, np.nan, Z)

    # Generate X, Y coordinate arrays
    transform = profile.get("transform", Affine.identity())
    ii, jj = np.meshgrid(
        np.arange(Z.shape[1]), np.arange(Z.shape[0])
    )  # x and y indices
    X, Y = rt.xy(transform, jj, ii)  # x and y coordinates
    X, Y = np.reshape(X, (-1,)).reshape(Z.shape), np.reshape(Y, (-1,)).reshape(Z.shape)

    return Z, X, Y, transform
