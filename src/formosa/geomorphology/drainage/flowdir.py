"""
Computes and analyse raster flow directions.

The analyses in this module operate on raster flow fields; explicit
flow-graph representations are implemented in :mod:`formosa.geomorphology.drainage.network`.

Last modified: 2026-08-22, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np

from formosa.utils import Backend, raise_fortran_error
from formosa.geomorphology._validation import (
    validate_same_shape,
    validate_format_dem,
    validate_format_valids,
    validate_format_flowdirs,
)
from formosa.geomorphology.drainage.directions import D8Directions
from formosa.geomorphology.drainage.preprocessing import fill_depressions
from formosa.geomorphology.drainage.flat_resolution import (
    label_flats,
    find_flat_edges,
    create_pulling_syn_grad,
    create_pushing_syn_grad,
    compute_syn_flowdir,
)
from formosa.geomorphology._native import drainage_flowdir as flowdir_f
import formosa.geomorphology.drainage._backends.flowdir_py as flowdir_py
from formosa.utils import NpFlowDir, NpReal

from typing import Optional
from numpy.typing import NDArray


def _compute_flowdir_simple(
    dem: NDArray[NpReal],
    dir_scheme: D8Directions = D8Directions(),
    valids: Optional[NDArray[np.bool_]] = None,
    backend: Backend = "fortran",
) -> tuple[NDArray[NpFlowDir], NDArray[np.bool_]]:
    """
    Computes flow directions for a DEM using a simple D8 algorithm.

    Parameters
    ----------
    dem : NDArray[number]
        A 2D array representing the digital elevation model (DEM).
    dir_scheme : D8Directions, optional
        An instance of `D8Directions` defining the flow direction scheme.
        Default is `D8Directions()`.
    valids : NDArray[bool], optional
        A boolean mask array indicating valid cells in the DEM.
        If `None`, all cells are considered valid.
        Default is `None`.
    backend : {'fortran', 'python'}, optional
        Backend to use for computation.
        `'fortran'` uses the Fortran extension for performance,
        while `'python'` uses a pure Python implementation.
        Default backend is `'fortran'`.

    Returns
    -------
    dirs : NDArray[uint8]
        A 2D integer array representing the flow directions for each cell in the DEM.
    flats : NDArray[bool]
        A boolean mask array where True indicates cells that are part of flat areas.
    """
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
    dir_scheme: D8Directions = D8Directions(),
    valids: Optional[NDArray[np.bool_]] = None,
    step_size: int = 4,
) -> tuple[NDArray[NpFlowDir], NDArray[np.bool_], NDArray[np.integer]]:
    """
    Computes flow directions for a DEM, resolving flat areas using synthetic elevations.
    Combines simple flow direction computation with flat area resolution from [R. Barnes *et al.* (2014)](https://doi.org/10.1016/j.cageo.2013.01.009).

    Parameters
    ----------
    dem : NDArray[number]
        A 2D array representing the digital elevation model (DEM).
    dir_scheme : D8Directions, optional
        An instance of `D8Directions` defining the flow direction scheme.
        Default is `D8Directions()`.
    valids : NDArray[bool], optional
        A boolean mask array indicating valid cells in the DEM.
        If `None`, all cells are considered valid.
        Default is `None`.
    step_size : int, optional
        The increment in synthetic elevation per step away from low edges to avoid ties when combined with the result of `compute_away_from_high`.
        Default is 4.

    Returns
    -------
    dirs : NDArray[uint8]
        A 2D integer array representing the flow directions for each cell in the DEM.
    flats : NDArray[bool]
        A boolean mask array where True indicates cells that are part of flat areas.
    z_syn : NDArray[int]
        A 2D integer array representing the synthetic elevation that resolves flat areas.
    """
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
    dir_scheme: D8Directions = D8Directions(),
    valids: Optional[NDArray[np.bool_]] = None,
    fill_depression: bool = False,
    resolve_flat: bool = True,
    step_size: int = 4,
) -> tuple[NDArray[NpFlowDir], NDArray[np.bool_], Optional[NDArray[np.integer]]]:
    """
    Computes flow directions for a DEM, optionally resolving flat areas.

    Parameters
    ----------
    dem : NDArray[number]
        2D array representing the digital elevation model (DEM).
    dir_scheme : D8Directions, optional
        Instance of `D8Directions` defining the flow direction scheme.
        Default is `D8Directions()`.
    valids : NDArray[bool], optional
        Boolean mask array indicating valid cells in the DEM.
        If `None`, all cells are considered valid.
        Default is `None`.
    fill_depression : bool, optional
        Whether to fill depressions in the DEM before computing flow directions.
        Default is False.
    resolve_flat : bool, optional
        Whether to resolve flat areas using synthetic elevations.
        Default is True.
    step_size : int, optional
        Increment in synthetic elevation per step away from low edges to avoid ties when combining synthetic elevations.
        Default is 4.

    Returns
    -------
    dirs : NDArray[uint8]
        2D integer array representing the flow directions for each cell in the DEM.
    flats : NDArray[bool]
        Boolean mask array where True indicates cells that are part of flat areas.
    syn_grads : NDArray[int] | None
        2D integer array representing the synthetic elevation that resolves flat areas, or None if `resolve_flat` is `False`.
    """
    dem = validate_format_dem(dem)
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
    dir_scheme: D8Directions = D8Directions(),
    valids: Optional[NDArray[np.bool_]] = None,
    backend: Backend = "fortran",
) -> NDArray[np.int8]:
    """
    Computes the number of upstream cells (in-degree) for each cell in a flow direction grid.

    Parameters
    ----------
    dirs : NDArray[uint8]
        2D array representing the flow directions for each cell
    dir_scheme : D8Directions, optional
        An instance of `D8Directions` defining the flow direction scheme
        Default is `D8Directions()`.
    valids : NDArray[int], optional
        2D array mask indicating whether the cell is valid
        If not provided, all cells are assumed to be valid.
        Default is `None`.
    backend : {'fortran', 'python'}, optional
        Backend to use for computation.
        `'fortran'` uses the Fortran extension for performance,
        while `'python'` uses a pure Python implementation.
        Default backend is `'fortran'`.

    Returns
    -------
    indegs : NDArray[int8]
        A 2D integer array representing the in-degree (number of upstream cells) for each cell.
    """
    dirs = validate_format_flowdirs(dirs)
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

    Raises
    ------
    RuntimeError
        If the traversal queue overflows or an unknown status is returned.
    MemoryError
        If the traversal workspace cannot be allocated.

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
    dir_scheme: D8Directions = D8Directions(),
    valids: Optional[NDArray[np.bool_]] = None,
    indegs: Optional[NDArray[np.integer]] = None,
    backend: Backend = "fortran",
) -> NDArray[np.bool_]:
    """
    Finds valid cells that do not belong to a directed flow cycle.

    Uses Kahn's algorithm to remove cells reachable from 0-in-degree cells.
    Valid cells remaining after the traversal belong to directed cycles.

    Parameters
    ----------
    dirs : NDArray[uint8]
        Flow directions for each cell.
    dir_scheme : D8Directions, optional
        Flow direction scheme defining the direction codes and offsets.
        The default scheme is `D8Directions()`.
    valids : NDArray[bool], optional
        Mask indicating cells included in the flow field. If `None`, all cells
        are considered valid.
        The default mask is `None`.
    indegs : NDArray[int], optional
        Indegrees computed for the same valid flow field. If `None`, they are
        computed using the selected backend.
        The default input is `None`.
    backend : {'fortran', 'python'}, optional
        Backend to use for computation.
        `'fortran'` uses the Fortran extension for performance,
        while `'python'` uses a pure Python implementation.
        Default backend is `'fortran'`.

    Returns
    -------
    acyclics : NDArray[bool]
        Mask that is true for valid acyclic cells and false for cyclic or
        invalid cells.

    Raises
    ------
    ValueError
        If an input shape or backend is invalid.
    MemoryError
        If the Fortran backend cannot allocate its workspace.
    RuntimeError
        If the Fortran backend reports queue overflow or an unexpected status.
    """
    dirs = validate_format_flowdirs(dirs)
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
    dir_scheme: D8Directions = D8Directions(),
    valids: Optional[NDArray[np.bool_]] = None,
    indegs: Optional[NDArray[np.integer]] = None,
    backend: Backend = "fortran",
) -> NDArray[np.bool_]:
    """
    Finds valid cells belonging to directed flow cycles.

    Parameters
    ----------
    dirs : NDArray[uint8]
        Flow directions for each cell.
    dir_scheme : D8Directions, optional
        Flow direction scheme defining the direction codes and offsets.
        The default scheme is `D8Directions()`.
    valids : NDArray[bool], optional
        Mask indicating cells included in the flow field. If `None`, all cells
        are considered valid.
        The default input is `None`.
    indegs : NDArray[int], optional
        Indegrees computed for the same valid flow field. If `None`, they are
        computed using the selected backend.
        The default input is `None`.
    backend : {'fortran', 'python'}, optional
        Backend to use for computation.
        `'fortran'` uses the Fortran extension for performance,
        while `'python'` uses a pure Python implementation.
        Default backend is `'fortran'`.

    Returns
    -------
    cyclics : NDArray[bool]
        Mask that is true for valid cyclic cells and false for acyclic or
        invalid cells.

    Raises
    ------
    ValueError
        If an input shape or backend is invalid.
    MemoryError
        If the Fortran backend cannot allocate its workspace.
    RuntimeError
        If the Fortran backend reports queue overflow or an unexpected status.
    """
    valids = validate_format_valids(valids, dirs, "flow direction raster")

    acyclics = find_acyclic_flowdirs(
        dirs,
        dir_scheme=dir_scheme,
        valids=valids,
        indegs=indegs,
        backend=backend,
    )
    return np.asarray(valids & ~acyclics, dtype=bool, order="F")
