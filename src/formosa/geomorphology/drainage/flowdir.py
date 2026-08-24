"""
Computes and analyses raster flow directions.

The analyses in this module operate on raster flow fields; explicit
flow-graph representations are implemented in
:mod:`formosa.geomorphology.drainage.network`.

Created: 2026-08-01, En-Chi Lee (williameclee@gmail.com)
Last modified: 2026-08-23, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np
from numpy.typing import NDArray

from formosa.geomorphology._native import drainage_flowdir as flowdir_f
from formosa.geomorphology.drainage._backends import flowdir_py
from formosa.geomorphology.drainage.directions import D8Directions
from formosa.geomorphology.drainage.flat_resolution import (
    compute_syn_flowdir,
    create_pulling_syn_grad,
    create_pushing_syn_grad,
    find_flat_edges,
    label_flats,
)
from formosa.geomorphology.drainage.preprocessing import fill_depressions
from formosa.geomorphology.raster_validation import (
    validate_format_dem,
    validate_format_dir_scheme,
    validate_format_flowdirs,
    validate_format_valids,
)
from formosa.utils import Backend, NpFlowDir, NpReal, raise_fortran_error
from formosa.utils.validation import validate_same_shape


def _compute_flowdir_simple(
    dem: NDArray[NpReal],
    dir_scheme: D8Directions | None = None,
    valids: NDArray[np.bool_] | None = None,
    backend: Backend = "fortran",
) -> tuple[NDArray[NpFlowDir], NDArray[np.bool_]]:
    """
    Computes flow directions for a DEM using a simple D8 algorithm.

    Parameters
    ----------
    dem : NDArray[number]
        Digital elevation model raster.
        - Expected shape: `(nrows, ncols)`.
    dir_scheme : D8Directions, optional
        Instance of `D8Directions` defining the flow direction scheme.
        - Default scheme is `D8Directions()`.
    valids : NDArray[bool], optional
        Boolean mask indicating valid cells in the DEM.
        If `None`, all cells are considered valid.
        - Expected shape: `(nrows, ncols)`, same as `dem`.
        - Default mask is `None`.
    backend : {'fortran', 'python'}, optional
        Backend to use for computation.
        `'fortran'` uses the Fortran extension for performance,
        while `'python'` uses a pure Python implementation.
        - Default backend is `'fortran'`.

    Returns
    -------
    dirs : NDArray[uint8]
        Flow direction raster.
        - Shape: `(nrows, ncols)`, same as `dem`.
    flats : NDArray[bool]
        Boolean mask indicating cells belonging to flat areas.
        - Shape: `(nrows, ncols)`, same as `dem`.
    """
    dir_scheme = validate_format_dir_scheme(dir_scheme)
    match backend:
        case "python":
            dirs, flats = flowdir_py.compute_flowdir_simple(dem, dir_scheme=dir_scheme)
        case "fortran":
            if valids is None:
                valids = np.ones(dem.shape, dtype=bool, order="F")
            dirs, flats = flowdir_f.compute_flowdir_simple(
                dem.astype(np.float32, order="F"),
                valids.astype(bool, order="F"),
                dir_scheme.offsets.astype(np.int32, order="F"),
                dir_scheme.codes.astype(np.uint8, order="F"),
            )
    return dirs.astype(np.uint8, order="F"), flats.astype(bool, order="F")


def _compute_flowdir_complete(
    dem: NDArray[NpReal],
    dir_scheme: D8Directions | None = None,
    valids: NDArray[np.bool_] | None = None,
    step_size: int = 4,
) -> tuple[NDArray[NpFlowDir], NDArray[np.bool_], NDArray[np.integer]]:
    """
    Computes flow directions for a DEM, resolving flat areas using
    synthetic elevations.

    Parameters
    ----------
    dem : NDArray[number]
        Digital elevation model raster.
        - Expected shape: `(nrows, ncols)`.
    dir_scheme : D8Directions, optional
        Instance of `D8Directions` defining the flow direction
        scheme.
        - Default scheme is `D8Directions()`.
    valids : NDArray[bool], optional
        Boolean mask indicating valid cells in the DEM.
        If `None`, all cells are considered valid.
        - Expected shape: `(nrows, ncols)`, same as `dem`.
        - Default mask is `None`.
    step_size : int, optional
        Increment in synthetic elevation per step away from low
        edges to avoid ties when combined with the result of
        :func:`compute_away_from_high`.
        - Default step size is 4.

    Returns
    -------
    dirs : NDArray[uint8]
        Flow direction raster.
        - Shape: `(nrows, ncols)`, same as `dem`.
    flats : NDArray[bool]
        Boolean mask indicating cells belonging to flat areas.
        - Shape: `(nrows, ncols)`, same as `dem`.
    z_syn : NDArray[int32]
        Synthetic elevation that resolves flat areas.
        - Shape: `(nrows, ncols)`, same as `dem`.

    Notes
    -----
    Combines simple flow direction computation with flat area
    resolution from [R Barnes *et al.* (2014)](https://doi.org/10.1016/j.cageo.2013.01.009).
    """
    dir_scheme = validate_format_dir_scheme(dir_scheme)
    if step_size <= 0:
        raise ValueError(f"Step size must be a positive integer (got {step_size}).")

    dirs, flats = _compute_flowdir_simple(dem, dir_scheme=dir_scheme, valids=valids)
    is_low_edge, is_high_edge = find_flat_edges(
        dem, dirs, dir_scheme=dir_scheme, valids=valids
    )
    flat_labels = label_flats(dem, (is_low_edge | flats), dir_scheme=dir_scheme)
    is_high_edge = is_high_edge & (flat_labels != 0)
    z_syn_away = create_pushing_syn_grad(
        flat_labels, is_high_edge, dir_scheme=dir_scheme
    )
    z_syn_towards = create_pulling_syn_grad(
        flat_labels,
        is_low_edge,
        dir_scheme=dir_scheme,
    )
    z_syn = z_syn_away + z_syn_towards * step_size

    flat_flowdir = compute_syn_flowdir(z_syn, flat_labels, dir_scheme=dir_scheme)
    dirs[dirs == 0] = flat_flowdir[dirs == 0]
    return dirs, flats, z_syn


def compute_flowdir(
    dem: NDArray[NpReal],
    dir_scheme: D8Directions | None = None,
    valids: NDArray[np.bool_] | None = None,
    fill_depression: bool = False,
    resolve_flat: bool = True,
    step_size: int = 4,
) -> tuple[NDArray[NpFlowDir], NDArray[np.bool_], NDArray[np.integer] | None]:
    """
    Computes flow directions for a DEM, optionally resolving flat areas.

    Parameters
    ----------
    dem : NDArray[number]
        Digital elevation model raster.
        - Expected shape: `(nrows, ncols)`.
    dir_scheme : D8Directions, optional
        Instance of `D8Directions` defining the flow direction scheme.
        - Default scheme is `D8Directions()`.
    valids : NDArray[bool], optional
        Boolean mask indicating valid cells in the DEM.
        If `None`, all cells are considered valid.
        - Expected shape: `(nrows, ncols)`, same as `dem`.
        - Default mask is `None`.
    fill_depression : bool, optional
        Whether to fill depressions in the DEM before computing flow
        directions.
        - Default option is `False`.
    resolve_flat : bool, optional
        Whether to resolve flat areas using synthetic elevations.
        - Default option is `True`.
    step_size : int, optional
        Increment in synthetic elevation per step away from low
        edges to avoid ties when combining synthetic elevations.
        - Default step size is 4.

    Returns
    -------
    dirs : NDArray[uint8]
        Flow direction raster.
        - Shape: `(nrows, ncols)`, same as `dem`.
    flats : NDArray[bool]
        Boolean mask indicating cells belonging to flat areas.
        - Shape: `(nrows, ncols)`, same as `dem`.
    syn_grads : NDArray[int32] | None
        Synthetic elevation that resolves flat areas, or `None` if
        `resolve_flat` is `False`.
        - Shape: `(nrows, ncols)`, same as `dem` (when present).
    """
    dem = validate_format_dem(dem)
    dir_scheme = validate_format_dir_scheme(dir_scheme)
    valids = validate_format_valids(valids, dem, "DEM")

    if fill_depression:
        dem = fill_depressions(dem, valids=valids)
    if resolve_flat:
        dirs, flats, syn_grads = _compute_flowdir_complete(
            dem, dir_scheme=dir_scheme, valids=valids, step_size=step_size
        )
    else:
        dirs, flats = _compute_flowdir_simple(dem, dir_scheme=dir_scheme, valids=valids)
        syn_grads = None
    return (
        dirs.astype(np.uint8, order="F"),
        flats.astype(bool, order="F"),
        syn_grads,
    )


def count_indegree(
    dirs: NDArray[NpFlowDir],
    dir_scheme: D8Directions | None = None,
    valids: NDArray[np.bool_] | None = None,
    backend: Backend = "fortran",
) -> NDArray[np.int8]:
    """
    Computes the number of upstream cells (in-degree) for each cell
    in a flow direction grid.

    Parameters
    ----------
    dirs : NDArray[uint8]
        Flow direction raster.
        - Expected shape: `(nrows, ncols)`.
    dir_scheme : D8Directions, optional
        Instance of `D8Directions` defining the flow direction
        scheme.
        - Default scheme is `D8Directions()`.
    valids : NDArray[bool], optional
        Boolean mask indicating valid cells in the flow direction
        grid.
        If `None`, all cells are considered valid.
        - Expected shape: `(nrows, ncols)`, same as `dirs`.
        - Default mask is `None`.
    backend : {'fortran', 'python'}, optional
        Backend to use for computation.
        `'fortran'` uses the Fortran extension for performance,
        while `'python'` uses a pure Python implementation.
        - Default backend is `'fortran'`.

    Returns
    -------
    indegs : NDArray[int8]
        In-degree (number of upstream cells) for each cell.
        - Shape: `(nrows, ncols)`, same as `dirs`.
    """
    dirs = validate_format_flowdirs(dirs)
    dir_scheme = validate_format_dir_scheme(dir_scheme)
    valids = validate_format_valids(valids, dirs, "flow direction raster")

    match backend:
        case "python":
            indegs = flowdir_py.count_indegree(
                dirs, dir_scheme=dir_scheme, valids=valids
            )
        case "fortran":
            indegs = flowdir_f.count_indegree(
                dirs.astype(np.uint8, order="F"),
                valids.astype(bool, order="F"),
                dir_scheme.offsets.astype(np.int32, order="F"),
                dir_scheme.codes.astype(np.uint8, order="F"),
            )

    return indegs.astype(np.int8, order="F")


def _find_acyclic_flowdirs_fortran(
    dirs: NDArray[NpFlowDir],
    indegs: NDArray[np.integer],
    valids: NDArray[np.bool_],
    dir_scheme: D8Directions,
) -> NDArray[np.bool_]:
    """
    Finds acyclic flow cells using the Fortran backend.

    Notes
    -----
    This is a helper function for :func:`find_acyclic_flowdirs`.
    """
    acyclics, err_code = flowdir_f.find_acyclic_flowdirs(
        dirs.astype(np.uint8, order="F"),
        indegs.astype(np.int8, order="F"),
        valids.astype(bool, order="F"),
        dir_scheme.offsets.astype(np.int32, order="F"),
        dir_scheme.codes.astype(np.uint8, order="F"),
    )
    raise_fortran_error("find_acyclic_flowdirs", err_code)
    return acyclics.astype(bool, order="F")


def find_acyclic_flowdirs(
    dirs: NDArray[NpFlowDir],
    dir_scheme: D8Directions | None = None,
    valids: NDArray[np.bool_] | None = None,
    indegs: NDArray[np.integer] | None = None,
    backend: Backend = "fortran",
) -> NDArray[np.bool_]:
    """
    Finds valid cells that do not belong to a directed flow cycle.

    Uses Kahn's algorithm to remove cells reachable from 0-in-degree
    cells. Valid cells remaining after the traversal belong to
    directed cycles.

    Parameters
    ----------
    dirs : NDArray[uint8]
        Flow direction raster.
        - Expected shape: `(nrows, ncols)`.
    dir_scheme : D8Directions, optional
        Flow direction scheme defining direction codes and offsets.
        - Default scheme is `D8Directions()`.
    valids : NDArray[bool], optional
        Boolean mask indicating valid cells in the flow field.
        If `None`, all cells are considered valid.
        - Expected shape: `(nrows, ncols)`, same as `dirs`.
        - Default mask is `None`.
    indegs : NDArray[int], optional
        In-degrees computed for the same valid flow field.
        If `None`, they are computed using the selected backend.
        - Expected shape: `(nrows, ncols)`, same as `dirs`.
        - Default input is `None`.
    backend : {'fortran', 'python'}, optional
        Backend to use for computation.
        `'fortran'` uses the Fortran extension for performance,
        while `'python'` uses a pure Python implementation.
        - Default backend is `'fortran'`.

    Returns
    -------
    acyclics : NDArray[bool]
        Boolean mask indicating valid acyclic cells.
        - Shape: `(nrows, ncols)`, same as `dirs`.
    """
    dirs = validate_format_flowdirs(dirs)
    dir_scheme = validate_format_dir_scheme(dir_scheme)
    valids = validate_format_valids(valids, dirs, "flow direction raster")

    if indegs is None:
        indegs = count_indegree(dirs, dir_scheme, valids=valids, backend=backend)
    validate_same_shape(dirs, indegs, "the flow direction", "the in-degree rasters")

    match backend:
        case "python":
            acyclics = flowdir_py.find_acyclic_flowdirs(
                dirs, indegs, valids, dir_scheme=dir_scheme
            )
        case "fortran":
            acyclics = _find_acyclic_flowdirs_fortran(dirs, indegs, valids, dir_scheme)

    return np.asarray(acyclics & valids, dtype=bool, order="F")


def find_cyclic_flowdirs(
    dirs: NDArray[NpFlowDir],
    dir_scheme: D8Directions | None = None,
    valids: NDArray[np.bool_] | None = None,
    indegs: NDArray[np.integer] | None = None,
    backend: Backend = "fortran",
) -> NDArray[np.bool_]:
    """
    Finds valid cells belonging to directed flow cycles.

    Parameters
    ----------
    dirs : NDArray[uint8]
        Flow direction raster.
        - Expected shape: `(nrows, ncols)`.
    dir_scheme : D8Directions, optional
        Flow direction scheme defining direction codes and offsets.
        - Default scheme is `D8Directions()`.
    valids : NDArray[bool], optional
        Boolean mask indicating valid cells in the flow field.
        If `None`, all cells are considered valid.
        - Expected shape: `(nrows, ncols)`, same as `dirs`.
        - Default mask is `None`.
    indegs : NDArray[int], optional
        In-degrees computed for the same valid flow field.
        If `None`, they are computed using the selected backend.
        - Expected shape: `(nrows, ncols)`, same as `dirs`.
        - Default input is `None`.
    backend : {'fortran', 'python'}, optional
        Backend to use for computation.
        `'fortran'` uses the Fortran extension for performance,
        while `'python'` uses a pure Python implementation.
        - Default backend is `'fortran'`.

    Returns
    -------
    cyclics : NDArray[bool]
        Boolean mask indicating valid cyclic cells.
        - Shape: `(nrows, ncols)`, same as `dirs`.
    """
    dirs = validate_format_flowdirs(dirs)
    dir_scheme = validate_format_dir_scheme(dir_scheme)
    valids = validate_format_valids(valids, dirs, "flow direction raster")

    acyclics = find_acyclic_flowdirs(
        dirs, dir_scheme, valids=valids, indegs=indegs, backend=backend
    )
    return np.asarray(valids & ~acyclics, dtype=bool, order="F")
