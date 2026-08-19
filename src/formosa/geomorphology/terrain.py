"""
Computes terrain metrics from digital elevation model rasters.

Last modified: 2026-08-19, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np

from formosa.geomorphology._native import terrain as terrain_f
from formosa.utils import Coords, NpCoords, raise_fortran_error

from typing import Optional
from numpy.typing import NDArray


def compute_slope(
    dem: NDArray[np.number],
    x: Optional[NDArray[NpCoords]] = None,
    y: Optional[NDArray[NpCoords]] = None,
    dx: Optional[Coords] = None,
    dy: Optional[Coords] = None,
) -> NDArray[NpCoords]:
    if x is not None:
        dxx = np.gradient(x, axis=1)
    elif dx is not None:
        dxx = dx
    else:
        dxx = 1
    if y is not None:
        dyy = np.gradient(y, axis=0)
    elif dy is not None:
        dyy = dy
    else:
        dyy = 1

    slope_y, slope_x = np.gradient(dem)
    slope_x /= dxx
    slope_y /= dyy

    slope = np.sqrt(slope_x**2 + slope_y**2)
    return slope


def calculate_isolation(
    dem: NDArray[np.number],
    valids: Optional[NDArray[np.bool_]] = None,
) -> (
    NDArray[np.float32]
    | tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.int32]]
):
    """
    Calculates terrain isolation within a raster neighbourhood.

    For each valid cell, the isolation is the grid distance to the
    nearest strictly higher cell selected by the native terrain
    backend.

    Parameters
    ----------
    dem : NDArray[number]
        Two-dimensional digital elevation model.
    valids : NDArray[bool], optional
        Boolean mask indicating valid cells. Non-finite DEM cells
        are always invalid.
        If `None`, all finite cells are valid.
        Default input is `None`.

    Returns
    -------
    isos : NDArray[float32]
        Isolation distances in grid-cell units, with the same shape
        as `dem`.
    ilpis, ilpjs : NDArray[int32], optional
        0-based row and column indices of the isolation limit points.
        Returned only when `return_ilp=True`.

    Raises
    ------
    ValueError
        If `dem` is empty or not two-dimensional, or if `valids`
        does not have the same shape as `dem`.
    TypeError
        If `dem` is not a real numeric array or `return_ilp` is not
        a boolean.
    RuntimeError
        If the FORTRAN backend reports an execution error.
    """
    dem_array = np.asarray(dem)
    if dem_array.ndim != 2 or 0 in dem_array.shape:
        raise ValueError(
            f"dem must be a non-empty 2D array, got shape {dem_array.shape}."
        )
    if not np.issubdtype(dem_array.dtype, np.number):
        raise TypeError(f"dem must have a numeric dtype, got {dem_array.dtype}.")
    if np.issubdtype(dem_array.dtype, np.complexfloating):
        raise TypeError("dem must contain real-valued elevations.")

    finite = np.isfinite(dem_array)
    if valids is None:
        valids_array = finite
    else:
        valids_array = np.asarray(valids, dtype=bool)
        if valids_array.shape != dem_array.shape:
            raise ValueError(
                f"Shapes for dem ({dem_array.shape}) and valids "
                f"({valids_array.shape}) do not match."
            )
        valids_array = valids_array & finite

    isos, ilpis, ilpjs, err_code = terrain_f.calculate_isolation(
        dem_array.astype(np.float32, order="F"),
        valids_array.astype(bool, order="F"),
    )
    raise_fortran_error("calculate_isolation", err_code)

    isos = np.asarray(isos, dtype=np.float32, order="F")

    ilpis = np.asarray(ilpis, dtype=np.int32, order="F")
    ilpjs = np.asarray(ilpjs, dtype=np.int32, order="F")
    has_ilp = (ilpis > 0) & (ilpjs > 0)
    ilpis = np.where(has_ilp, ilpis - 1, -1).astype(np.int32, order="F", copy=False)
    ilpjs = np.where(has_ilp, ilpjs - 1, -1).astype(np.int32, order="F", copy=False)
    return isos, ilpis, ilpjs
