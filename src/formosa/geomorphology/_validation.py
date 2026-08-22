"""
Validates function input arguments.

Created: 2026-08-22, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np

from formosa.utils import NpReal, NpFlowDir

from typing import Optional
from numpy.typing import NDArray


def validate_same_shape(
    arr1: NDArray, arr2: NDArray, arr1_name: str = "array 1", arr2_name: str = "array 2"
):
    if arr1.shape != arr2.shape:
        raise ValueError(
            f"Shapes for {arr1_name} and {arr2_name} must match, "
            f"but got shapes {arr1.shape} and {arr2.shape}, respectively."
        )


def validate_format_dem(dem: NDArray[NpReal]) -> NDArray[NpReal]:
    dem = np.asarray(dem)
    if dem.ndim != 2 or 0 in dem.shape:
        raise ValueError(
            "DEM must be a non-empty 2D array, " + f"but received shape {dem.shape}."
        )
    if not np.issubdtype(dem.dtype, np.number):
        raise TypeError("DEM must have a numeric dtype, " + f"but got {dem.dtype}.")
    if np.issubdtype(dem.dtype, np.complexfloating):
        raise TypeError(
            "DEM must contain real-valued elevations, " + f"but got type {dem.dtype}."
        )
    return dem


def validate_format_valids(
    valids: Optional[NDArray[np.bool_]],
    against: Optional[NDArray],
    against_name: str = "masked array",
) -> NDArray[np.bool_]:
    if against is None and valids is not None:
        if valids.ndim != 2 or 0 in valids.shape:
            raise ValueError(
                "Valids must be a non-empty 2D array when supplied, "
                + f"but received shape {valids.shape}."
            )
        return valids
    if against is None:
        raise ValueError("Cannot determine what the validity mask should be.")
    finite = np.isfinite(against)
    if valids is None:
        return finite
    valids = np.asarray(valids, dtype=bool)
    validate_same_shape(valids, against, "validity mask", against_name)
    valids = valids & finite
    return valids


def validate_format_flowdirs(
    dirs: NDArray[NpFlowDir],
    against: Optional[NDArray] = None,
    against_name: str = "DEM",
) -> NDArray[NpFlowDir]:
    dirs = np.asarray(dirs)
    if against is None:
        if dirs.ndim != 2 or 0 in dirs.shape:
            raise ValueError(
                "Flow direction raster must be a non-empty 2D array, "
                + f"but received shape {dirs.shape}."
            )
    else:
        validate_same_shape(dirs, against, "flow direction raster", against_name)
    if not np.issubdtype(dirs.dtype, np.integer):
        raise TypeError(
            "Flow direction raster must have uint8 dtype, " + f"but got {dirs.dtype}."
        )
    if np.isdtype(dirs.dtype, NpFlowDir):
        return dirs
    if (np.min(dirs) < 0) or (np.max(dirs) >= 255):
        raise TypeError(
            "Flow direction values must be in the range [0, 255], "
            + f"but got range [{np.min(dirs)}, {np.max(dirs)}]."
        )
    return dirs.astype(NpFlowDir)
