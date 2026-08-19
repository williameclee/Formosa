"""Computes slope and isolation from gridded elevation data.

This module exposes public NumPy APIs for terrain metrics. Isolation
uses the internal Fortran backend. Results include nearest-higher
cells and outer-boundary censoring information.

Last modified: 2026-08-19, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np

from formosa.geomorphology._native import terrain as terrain_f
from formosa.utils import Coords, NpCoords, NpCanonIndex, raise_fortran_error

from typing import Optional
from numpy.typing import NDArray


def compute_slope(
    dem: NDArray[np.number],
    x: Optional[NDArray[NpCoords]] = None,
    y: Optional[NDArray[NpCoords]] = None,
    dx: Optional[Coords] = None,
    dy: Optional[Coords] = None,
) -> NDArray[NpCoords]:
    """
    Calculates *slope magnitude* from a gridded DEM.

    Horizontal derivatives are scaled using coordinate arrays when
    supplied, otherwise by constant grid spacing. Coordinate arrays
    take precedence over the corresponding scalar spacing.

    Parameters
    ----------
    dem : NDArray[number]
        2D digital elevation model.
    x, y : NDArray[float], optional
        Horizontal coordinate arrays for DEM columns and rows,
        respectively.
        Their gradients define local grid spacing.
        Default inputs are `None`.
    dx, dy : int | float, optional
        Constant column and row spacing, respectively.
        Each defaults to `1` when neither it nor its coordinate
        array is supplied.
        Default spacings are `None`.

    Returns
    -------
    slope : NDArray[float]
        Magnitude of the elevation gradient in rise per horizontal
        distance unit, with the same shape as `dem`.
    """
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


def compute_isolation(
    dem: NDArray[np.number],
    valids: Optional[NDArray[np.bool_]] = None,
    dx: Coords = 1.0,
    dy: Coords = 1.0,
) -> tuple[
    NDArray[np.float32], NDArray[NpCanonIndex], NDArray[NpCanonIndex], NDArray[np.bool_]
]:
    """
    Calculates terrain *isolation* and boundary censoring for a DEM.

    For each cell, isolation is the physical distance to the
    nearest strictly higher cell (the isolation limiting point,
    ILP). Highest cells have no ILP and receive isolation 0 and ILP
    indices of -1.

    Parameters
    ----------
    dem : NDArray[number]
        2D digital elevation model.
    valids : NDArray[bool], optional
        Boolean mask indicating valid cells.
        Non-finite DEM cells are always invalid.
        If `None`, all finite cells are assumed valid.
        Default input is `None`.
    dx, dy : int | float, optional
        Positive, finite column and row spacing, respectively.
        The isolation distances use the same units as these values.
        Both default to `1.0`.

    Returns
    -------
    isos : NDArray[float32]
        Isolation distances in the units of `dx` and `dy`, with the
        same shape as `dem`.
        Invalid cells and cells without an ILP contain `0`.
    ilpis, ilpjs : NDArray[int32]
        0-based row (*y*) and column (*x*) indices of the isolation
        limit points.
        Cells without an ILP and invalid cells contain `-1`.
    censored : NDArray[bool]
        Whether the isolation search reaches beyond the outer raster
        footprint before it reaches the reported ILP. 
        Valid cells with no ILP are censored; invalid cells are 
        not. Internal invalid regions are not treated as 
        observation-window boundaries.

    Raises
    ------
    ValueError
        If `dem` is empty or not two-dimensional, or if `valids`
        does not have the same shape as `dem`, or if either grid
        spacing is non-finite, non-positive, or cannot be
        represented by the native backend.
    TypeError
        If `dem` is not a real numeric array or either grid spacing
        is not a real numeric scalar.
    RuntimeError
        If the Fortran backend reports an execution error.

    Notes
    -----
    The DEM is assumed a regular, axis-aligned grid whose footprint
    extends half a cell beyond each outer cell centre. Internal
    invalid cells are excluded as ILPs.
    """
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

    spacings: dict[str, np.float32] = {}
    for name, spacing in (("dx", dx), ("dy", dy)):
        spacing_array = np.asarray(spacing)
        if (
            isinstance(spacing, (bool, np.bool_))
            or spacing_array.ndim != 0
            or not np.issubdtype(spacing_array.dtype, np.number)
            or np.issubdtype(spacing_array.dtype, np.complexfloating)
        ):
            raise TypeError(f"{name} must be a real numeric scalar.")

        spacing_value = float(spacing_array)
        if not np.isfinite(spacing_value) or spacing_value <= 0.0:
            raise ValueError(
                f"{name} must be finite and greater than 0, " + f"but got {spacing}."
            )
        if spacing_value > np.finfo(np.float32).max:
            raise ValueError(f"{name} is too large for the native backend: {spacing}.")
        spacings[name] = np.float32(spacing_value)

    dx_f = spacings["dx"]
    dy_f = spacings["dy"]

    finite = np.isfinite(dem)
    if valids is None:
        valids = finite
    else:
        valids = np.asarray(valids, dtype=bool)
        if valids.shape != dem.shape:
            raise ValueError(
                "Shapes for DEM and validity mask must match, "
                f"but got shapes {dem.shape} and {valids.shape}, respectively."
            )
        valids = valids & finite

    isos, ilpis, ilpjs, censored, err_code = terrain_f.compute_isolation(
        dem.astype(np.float32, order="F"),
        valids.astype(bool, order="F"),
        dx_f,
        dy_f,
    )
    raise_fortran_error("compute_isolation", err_code)

    isos = np.asarray(isos, dtype=np.float32, order="F")

    ilpis = np.asarray(ilpis, dtype=NpCanonIndex, order="F")
    ilpjs = np.asarray(ilpjs, dtype=NpCanonIndex, order="F")
    has_ilp = (ilpis > 0) & (ilpjs > 0)
    ilpis = np.where(has_ilp, ilpis - 1, -1).astype(NpCanonIndex, order="F", copy=False)
    ilpjs = np.where(has_ilp, ilpjs - 1, -1).astype(NpCanonIndex, order="F", copy=False)
    censored = np.asarray(censored, dtype=bool, order="F")
    return isos, ilpis, ilpjs, censored
