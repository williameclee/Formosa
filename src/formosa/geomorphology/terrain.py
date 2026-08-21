"""
Computes slope, isolation, and prominence from gridded elevation
data.

This module exposes public NumPy APIs for terrain metrics. Isolation
and prominence use the internal Fortran backend.

Last modified: 2026-08-21, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np

from formosa.geomorphology.drainage.directions import D8Directions
from formosa.geomorphology._native import terrain as terrain_f
from formosa.utils import NpReal, Coords, NpCoords, NpCanonIndex, raise_fortran_error

from typing import Optional
from numpy.typing import NDArray


def _validate_format_dem(dem: NDArray[NpReal]) -> NDArray[NpReal]:
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


def _validate_format_valids(
    valids: Optional[NDArray[np.bool_]], dem: NDArray[NpReal]
) -> NDArray[np.bool_]:
    finite = np.isfinite(dem)
    if valids is None:
        return finite
    valids = np.asarray(valids, dtype=bool)
    if valids.shape != dem.shape:
        raise ValueError(
            "Shapes for DEM and validity mask must match, "
            f"but got shapes {dem.shape} and {valids.shape}, respectively."
        )
    valids = valids & finite
    return valids  # type: ignore


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
    See [Kirmse & de Ferranti (2017)](https://doi.org/10.1177/0309133317738163)
    for the definition of isolation and more details.

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


def compute_prominence(
    dem: NDArray[NpReal],
    valids: Optional[NDArray[np.bool_]] = None,
    dir_scheme: D8Directions = D8Directions(),
) -> tuple[
    NDArray[NpReal | np.int64],
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.int32],
]:
    """
    Calculates *topographic prominence* and its divide-tree 
    features.

    Prominence measures the vertical height of a summit relative to
    the highest key saddle connecting it to higher terrain. It is
    computed using a descending topological sweep-plane algorithm.

    Parameters
    ----------
    dem : NDArray[number]
        2D digital elevation model.
    valids : NDArray[bool], optional
        Boolean mask indicating valid cells.
        Non-finite DEM cells are always invalid.
        If `None`, all finite cells are assumed valid.
        Default input is `None`.
    dir_scheme : D8Directions, optional
        Direction scheme defining neighbour connectivity offsets.
        Default scheme is `D8Directions()`.

    Returns
    -------
    proms : NDArray[number | int64]
        Topographic prominence heights with the same shape and
        numeric type as `dem` (or `int64` if `dem` is unsigned
        integer).
        Subordinate summits contain their prominence above the key
        saddle. Non-summit cells contain `0`. Invalid cells and the
        highest regional summits with unknown prominence contain
        `-1`.
    feats : NDArray[int32]
        0-based feature index at peak and saddle cells, with `-1` at
        cells that do not represent a feature.
    feat_types : NDArray[int32]
        Feature type indexed by feature: `1` for a peak and `2` for
        a saddle.
    key_saddles : NDArray[int32]
        Key-saddle feature index for each peak feature. Entries for
        non-peak features and peaks without a known key saddle are
        `-1`.
    feat_prnts : NDArray[int32]
        Parent feature index for each feature in the divide tree.
        Root features contain `-1`.

    Raises
    ------
    ValueError
        If `dem` is empty or not 2D, or if `valids` does not have
        the same shape as `dem`.
    TypeError
        If `dem` is not a real numeric array.
    RuntimeError
        If the Fortran backend reports an execution error.

    Notes
    -----
    See [Kirmse & de Ferranti (2017)](https://doi.org/10.1177/0309133317738163)
    for the definition of prominence and more details.
    """
    dem = _validate_format_dem(dem)
    valids = _validate_format_valids(valids, dem)

    dem_f = np.asfortranarray(dem, dtype=np.float32)
    valids_f = np.asfortranarray(valids, dtype=bool)

    valid_ids = np.flatnonzero(valids_f.ravel(order="F"))
    orders = np.argsort(dem_f.ravel(order="F")[valid_ids])

    # 1-based IDs in ascending elevation order.
    orders_f = valid_ids[orders].astype(np.int32) + 1

    proms, feats, feat_types, key_saddles, feat_prnts, nfeats, err_code = (
        terrain_f.compute_prominence(
            dem_f, orders_f, dir_scheme.offsets.astype(np.int32, order="F")
        )
    )
    raise_fortran_error("compute_prominence", err_code)

    # Try to make `proms` the same type as `dem`, unless `dem` is
    # unsigned because `proms` needs `-1` to mark invalid cells and
    # the highest peaks with unknown prominence
    if not np.issubdtype(dem.dtype, np.unsignedinteger):
        proms = np.asarray(proms, dtype=dem.dtype)
    else:
        proms = np.asarray(proms, dtype=np.int64)

    proms[~valids] = -1

    # Convert 1-based Fortran IDs with zero sentinels into zero-based
    # Python indices with -1 sentinels.
    feats = np.asarray(feats, dtype=np.int32) - 1
    feat_types = np.array(feat_types[:nfeats], dtype=np.int32, copy=True)
    key_saddles = np.array(key_saddles[:nfeats], dtype=np.int32, copy=True) - 1
    feat_prnts = np.array(feat_prnts[:nfeats], dtype=np.int32, copy=True) - 1

    return proms, feats, feat_types, key_saddles, feat_prnts
