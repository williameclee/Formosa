"""
Validates and post-processes downloaded digital elevation model
data.

This module provides internal helpers shared by the DEM download
APIs and is not intended to be used directly.

Last modified: 2026-08-23, En-Chi Lee (williameclee@gmail.com)
"""

import warnings

import numpy as np
from numpy.typing import NDArray
from rasterio.transform import Affine

from formosa.utils import Real


def _validate_latlon_limits(
    latlim: tuple[Real, Real], lonlim: tuple[Real, Real], format: bool = True
) -> tuple[tuple[Real, Real], tuple[Real, Real]]:
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


def _dem_post_processing(
    Z: NDArray[np.floating | np.integer], profile: dict
) -> tuple[
    NDArray[np.floating | np.integer],
    NDArray[np.floating],
    NDArray[np.floating],
    Affine,
]:
    """
    Common DEM post-processing steps.
    """
    from formosa.dem.utils import transform2xy

    # Replace no data values with NaN
    no_data_value = profile.get("nodata", None)
    if no_data_value is not None:
        Z = np.where(Z == no_data_value, np.nan, Z)

    # Generate X, Y coordinate arrays
    transform = profile.get("transform", Affine.identity())
    X, Y = transform2xy(transform, Z.shape)

    return Z, X, Y, transform
