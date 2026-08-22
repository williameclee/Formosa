"""
Prepares digital elevation models for drainage analysis.

This module identifies ocean basins, fills depressions, and performs
other operations required before flow routing and metric
calculation.

Last modified: 2026-08-22, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np

from formosa.geomorphology._validation import (
    validate_format_dem,
    validate_format_valids,
)
from formosa.utils import raise_fortran_error
from formosa.geomorphology.drainage.directions import D8Directions
from formosa.geomorphology._native import drainage_preprocessing as preproc_f
from formosa.utils import NpReal

from typing import Optional
import numpy.typing as npt


def detect_ocean_basins_from_boundary(
    dem: npt.NDArray[NpReal],
    valids: Optional[npt.NDArray[np.bool_]] = None,
    ocean_level: int | float = 0,
    flood_below: bool = True,
    dir_scheme: D8Directions = D8Directions(),
) -> npt.NDArray[np.int32]:
    """
    Labels threshold-matching ocean basins connected to the raster boundary.

    A nonzero label identifies one connected boundary basin. Invalid cells, cells above `ocean_level`, and qualifying cells disconnected from the boundary receive label zero.
    When `flood_below` is false, only cells exactly equal to `ocean_level` are included.

    Parameters
    ----------
    dem : NDArray[number]
        2D digital elevation model.
    valids : NDArray[bool], optional
        Boolean mask indicating valid cells.
        Invalid cells are excluded from ocean basin detection. If `None`, every cell with a finite elevation is valid.
        Default input is `None`.
    ocean_level : int | float, optional
        Elevation threshold defining ocean cells.
        Default value is `0`.
    flood_below : bool, optional
        Whether elevations strictly below `ocean_level` qualify as ocean cells.
        When false, only cells exactly equal to `ocean_level` qualify.
        Default option is `True`.
    dir_scheme : D8Directions, optional
        Instance of `D8Directions` defining neighbour offsets.
        Default scheme is `D8Directions()`.

    Returns
    -------
    NDArray[int32]
        Ocean basin labels with the same shape as `dem`.

    Raises
    ------
    ValueError
        If `dem` is empty or not two-dimensional, or if `valids` shape does not
        match `dem`.
    TypeError
        If `dem` does not have a numeric dtype.
    RuntimeError
        If the Fortran routine encounters an execution error.

    Notes
    -----
    See also: :func:`invalidate_ocean_basins`
    """
    dem = validate_format_dem(dem)
    valids = validate_format_valids(valids, dem, "DEM")

    if isinstance(ocean_level, (bool, np.bool_)) or not np.isscalar(ocean_level):
        raise TypeError("ocean_level must be a real numeric scalar.")
    try:
        ocean_lvl_float = float(ocean_level)  # type: ignore
    except (TypeError, ValueError) as exc:
        raise TypeError("ocean_level must be a real numeric scalar.") from exc
    if not np.isfinite(ocean_lvl_float):
        raise ValueError("ocean_level must be finite.")
    if not isinstance(flood_below, (bool, np.bool_)):
        raise TypeError("flood_below must be a boolean.")

    if not np.any(valids):
        return np.zeros(dem.shape, dtype=np.int32, order="F")

    basins, err_code = preproc_f.detect_ocean_basins_from_boundary(
        dem.astype(np.float32, order="F"),
        valids.astype(bool, order="F"),
        dir_scheme.offsets.astype(np.int32, order="F"),
        np.float32(ocean_lvl_float),
        bool(flood_below),
    )
    raise_fortran_error("detect_ocean_basins_from_boundary", err_code)
    return np.asarray(basins, dtype=np.int32)


def invalidate_ocean_basins(
    dem: npt.NDArray[NpReal],
    valids: Optional[npt.NDArray[np.bool_]] = None,
    ocean_level: int | float = 0,
    flood_below: bool = True,
    min_size: int = 1,
    dir_scheme: D8Directions = D8Directions(),
) -> npt.NDArray[np.bool_]:
    """
    Returns a validity mask with sufficiently large ocean basins
    invalidated.

    Parameters
    ----------
    dem : NDArray[number]
        2D digital elevation model.
    valids : NDArray[bool], optional
        Boolean mask indicating valid cells.
        Invalid cells remain invalid in the output mask. If `None`,
        every cell with a finite elevation is valid initially.
        Default input is `None`.
    ocean_level : int | float, optional
        Elevation threshold defining ocean cells.
        Default elevation is `0`.
    flood_below : bool, optional
        Whether elevations strictly below `ocean_level` qualify as ocean cells.
        When false, only cells exactly equal to `ocean_level` qualify.
        Default option is `True`.
    min_size : int, optional
        Minimum cell count threshold for ocean basin invalidation.
        Ocean basins containing at least this number of cells are invalidated.
        Default size is `1`.
    dir_scheme : D8Directions, optional
        Instance of `D8Directions` defining neighbour offsets.
        Default scheme is `D8Directions()`.

    Returns
    -------
    valids : NDArray[bool]
        Validity mask with sufficiently large ocean basin cells *also* set to `False`.

    Raises
    ------
    ValueError
        If `min_size` is less than 1, `dem` is empty or not 2D, or if `valids` shape does not match `dem`.
    TypeError
        If `dem` does not have a numeric dtype.
    RuntimeError
        If the Fortran routine encounters an execution error.

    Notes
    -----
    Ocean basins are detected using :func:`detect_ocean_basins_from_boundary`.
    Basins with cell counts smaller than `min_size` or disconnected from the boundary remain valid.
    """

    if isinstance(min_size, (bool, np.bool_)) or not isinstance(
        min_size, (int, np.integer)
    ):
        raise TypeError("min_size must be an integer.")
    if min_size < 1:
        raise ValueError("min_size must be at least 1.")

    dem = np.asarray(dem)
    basins = detect_ocean_basins_from_boundary(
        dem,
        valids=valids,
        ocean_level=ocean_level,
        flood_below=flood_below,
        dir_scheme=dir_scheme,
    )
    if valids is None:
        out_valids = np.isfinite(dem)
    else:
        out_valids = np.asarray(valids, dtype=bool).copy()
        out_valids &= np.isfinite(dem)

    counts = np.bincount(basins.ravel())
    sufficiently_large = counts >= min_size
    sufficiently_large[0] = False
    out_valids[sufficiently_large[basins]] = False
    return out_valids


def fill_depressions(
    dem: npt.NDArray[NpReal],
    dir_scheme: D8Directions = D8Directions(),
    valids: Optional[npt.NDArray[np.bool_]] = None,
    max_fill_size: Optional[int] = None,
) -> npt.NDArray[NpReal]:
    """
    Fills depressions in a digital elevation model (DEM).

    Interior cells that cannot drain to the edge of the array are
    raised to the lowest elevation that provides an outlet. If an
    upper size limit for a depression is set, sufficiently large
    depressions are considered internally-drained basins, and the
    lowest point in each such basin becomes a priority-flood outlet.

    Parameters
    ----------
    dem : NDArray[number]
        2D DEM.
        The calculation uses 32-bit floating-point precision and
        converts the result back to the input dtype; the input is
        not modified.
    dir_scheme : D8Directions, optional
        Flow direction encoding scheme.
        Default scheme is `D8Directions()`.
    valids : NDArray[bool], optional
        Boolean mask with the same shape as `dem`.
        Invalid cells are excluded from the fill and retain their
        original elevations. Valid cells on the outer array boundary
        or adjacent to invalid cells are treated as priority-flood
        outlets.
        If `None`, every cell is assumed to be valid.
        Default input is `None`.
    max_fill_size : int, optional
        Maximum size (in cells) of a depression before it is
        considered an internally-drained basin instead.
        If `None`, all depressions are filled (equivalent to
        infinity).
        Default size is `None`.

    Returns
    -------
    dem_filled : NDArray[number]
        Depression-filled DEM.

    Notes
    -----
    Elevations should be finite. `NaN` ordering is not defined by
    the Fortran priority queue. Equal-elevation cells may be
    processed in any order without changing the filled result.
    """
    dem = validate_format_dem(dem)
    valids = validate_format_valids(valids, dem, "DEM")
    if not np.any(valids):
        return dem.copy()
    # Validate max_fill_size
    if max_fill_size and (max_fill_size < 0):
        raise ValueError(
            "Maximum fill size must be a non-negative integer, "
            + f"got {max_fill_size}."
        )

    dem_f32 = dem.astype(np.float32, order="F")
    dem_filled, err_code = preproc_f.fill_depressions(
        dem_f32,
        valids.astype(bool, order="F"),
        np.zeros(dem.shape, dtype=bool, order="F"),
        dir_scheme.offsets.astype(np.int32, order="F"),
    )
    raise_fortran_error("fill_depression", err_code)

    if max_fill_size is None:
        return dem_filled.astype(dem.dtype, order="F")

    # Find depressions
    labels, err_code = preproc_f.label_mask_areas(
        (valids & (dem_filled > dem_f32)).astype(bool, order="F"),
        dir_scheme.offsets.astype(np.int32, order="F"),
    )
    raise_fortran_error("label_mask_areas", err_code)
    # Count depressions
    counts = np.bincount(labels.ravel())
    is_large = counts > max_fill_size
    is_large[0] = False  # The first is 0, the non-depression cells
    # Treat large depressions as internally drained basins:
    # Add the lowest original cell of each basin to a shared sink
    # mask
    more_sinks = np.zeros(dem.shape, dtype=bool, order="F")
    for ibasin in np.flatnonzero(is_large):
        basin_mask = labels == ibasin
        basin_indices = np.flatnonzero(basin_mask)
        sink_index = basin_indices[np.argmin(dem_f32.ravel()[basin_indices])]
        more_sinks[np.unravel_index(sink_index, dem.shape)] = True

    # One more priority-flood pass
    if np.any(more_sinks):
        new_dem_filled, err_code = preproc_f.fill_depressions(
            dem_f32,
            valids.astype(bool, order="F"),
            more_sinks,
            dir_scheme.offsets.astype(np.int32, order="F"),
        )
        raise_fortran_error("fill_depression", err_code)
        large_basin_mask = is_large[labels]
        dem_filled[large_basin_mask] = new_dem_filled[large_basin_mask]
    return dem_filled.astype(dem.dtype, order="F")
