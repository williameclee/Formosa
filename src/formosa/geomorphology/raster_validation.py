"""
Validates raster formats for geomorphological operations.

This module implements internal validation routines for digital
elevation models, validity masks, flow directions, and coordinate
rasters.

Created: 2026-08-22, En-Chi Lee (williameclee@gmail.com)
Last modified: 2026-08-23, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np
from numpy.typing import NDArray

from formosa.utils import NpCoords, NpFlowDir, NpReal
from formosa.utils.validation import (
    validate_2d_raster,
    validate_same_shape,
    validate_shape,
)


def validate_format_dem(dem: NDArray[NpReal]) -> NDArray[NpReal]:
    """
    Validates and standardises a digital elevation model raster.

    Parameters
    ----------
    dem : NDArray[number]
        Digital elevation model raster.
        - Expected shape: `(nrows, ncols)`.

    Returns
    -------
    dem : NDArray[number]
        Validated digital elevation model raster.
        - Shape: `(nrows, ncols)`.
    """
    dem = np.asarray(dem)
    validate_2d_raster(dem, "DEM")
    if not np.issubdtype(dem.dtype, np.number):
        raise TypeError("DEM must have a numeric dtype, " + f"but got {dem.dtype}.")
    if np.issubdtype(dem.dtype, np.complexfloating):
        raise TypeError(
            "DEM must contain real-valued elevations, " + f"but got type {dem.dtype}."
        )
    return dem


def validate_format_valids(
    valids: NDArray[np.bool_] | None,
    against: NDArray[np.number | np.bool_] | None,
    against_name: str = "masked array",
) -> NDArray[np.bool_]:
    """
    Validates and normalises a boolean validity mask against a
    raster.

    Parameters
    ----------
    valids : NDArray[bool] | None
        Boolean mask indicating valid cells.
        If `None`, every cell in `against` with a finite value is
        valid.
        - Expected shape: `(nrows, ncols)`, same as `against` (when
            present).
    against : NDArray[number | bool] | None
        Reference array to compare the mask shape and finite values
        against.
        - Expected shape: `(nrows, ncols)`, same as `valids` (when
            present).
    against_name : str, optional
        Name of the reference array for error messages.
        - Default name is `'masked array'`.

    Returns
    -------
    valids : NDArray[bool]
        Boolean validity mask with non-finite reference cells marked
        invalid.
        - Shape: `(nrows, ncols)`, same as `against` (or `valids`).
    """
    if against is None and valids is not None:
        valids = np.asarray(valids, dtype=bool)
        validate_2d_raster(valids, "Supplied validity mask")
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
    against: NDArray[np.number | np.bool_] | None = None,
    against_name: str = "DEM",
) -> NDArray[NpFlowDir]:
    """
    Validates and standardises a flow direction raster.

    Parameters
    ----------
    dirs : NDArray[uint8]
        Flow direction raster.
        - Expected shape: `(nrows, ncols)`.
    against : NDArray[number | bool], optional
        Reference array to compare the flow direction raster shape
        against.
        If provided, `dirs` must match its shape.
        - Expected shape: `(nrows, ncols)`, same as `dirs`.
        - Default input is `None`.
    against_name : str, optional
        Name of the reference array for error messages.
        - Default name is `'DEM'`.

    Returns
    -------
    dirs : NDArray[uint8]
        Validated flow direction raster with `uint8` dtype.
        - Shape: `(nrows, ncols)`.
    """
    dirs = np.asarray(dirs)
    validate_2d_raster(dirs, "flow direction raster")
    if against is not None:
        validate_same_shape(dirs, against, "flow direction raster", against_name)
    if not np.issubdtype(dirs.dtype, np.integer):
        raise TypeError(
            "Flow direction raster must have uint8 dtype, " + f"but got {dirs.dtype}."
        )
    if np.isdtype(dirs.dtype, NpFlowDir):
        return dirs
    if (np.min(dirs) < 0) or (np.max(dirs) > 255):
        raise ValueError(
            "Flow direction values must be in the range [0, 255], "
            + f"but got range [{np.min(dirs)}, {np.max(dirs)}]."
        )
    return dirs.astype(NpFlowDir)


def validate_format_freeform_coordinates(
    x: NDArray[NpCoords] | None,
    y: NDArray[NpCoords] | None,
    shape: tuple[int, int] | None = None,
    dtype: type | None = None,
) -> tuple[NDArray[NpCoords], NDArray[NpCoords]]:
    """
    Validates or generates 2D coordinate rasters.

    Parameters
    ----------
    x : NDArray[number], optional
        Horizontal coordinate array.
        If `None`, column indices are generated using `shape`.
        - Expected shape: `(nrows, ncols)`, same as `shape` (when
            present).
        - Default input is `None`.
    y : NDArray[number], optional
        Vertical coordinate array.
        If `None`, row indices are generated using `shape`.
        - Expected shape: `(nrows, ncols)`, same as `shape` (when
            present).
        - Default input is `None`.
    shape : tuple[int, int], optional
        Expected raster shape `(nrows, ncols)`.
        Required when `X` and `Y` are not supplied.
        - Default shape is `None`.
    dtype : type, optional
        Target data type for the generated coordinate arrays.
        - Default dtype is `None`.

    Returns
    -------
    x : NDArray[number]
        Validated or generated horizontal coordinate raster.
        - Shape: `(nrows, ncols)`.
    y : NDArray[number]
        Validated or generated vertical coordinate raster.
        - Shape: `(nrows, ncols)`.
    """
    if x is not None and y is not None:
        if shape is not None:
            validate_shape(x, shape, "X coordinates")
            validate_shape(y, shape, "Y coordinates")
        else:
            validate_same_shape(x, y, "X coordinates", "Y coordinates")
        return x, y
    elif shape is None:
        raise ValueError("Cannot infer the shape of the coordinate rasters.")
    if x is not None:
        validate_shape(x, shape, "X coordinates")
    if y is not None:
        validate_shape(y, shape, "Y coordinates")

    x_lin = np.arange(shape[1])
    y_lin = np.arange(shape[0])
    x_grid, y_grid = np.meshgrid(x_lin, y_lin, indexing="xy")
    x = x if x is not None else np.asarray(x_grid, dtype=dtype)
    y = y if y is not None else np.asarray(y_grid, dtype=dtype)
    return x, y


__all__ = [
    "validate_format_dem",
    "validate_format_flowdirs",
    "validate_format_freeform_coordinates",
    "validate_format_valids",
]
