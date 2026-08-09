# Last modified
#   2026-02-11, En-Chi Lee (williameclee@arizona.edu)
#     - Rename flowdir functions to be more descriptive.
#   2026-06-09, En-Chi Lee (williameclee@gmail.com)
#     - Added error for missing FORTRAN backend.
#     - Removed NumPy type `np.bool` to either `np.bool_` or `bool`
#       for compatibility with newer NumPy versions.
#   2026-06-10, En-Chi Lee (williameclee@gmail.com)
#     - Small refactors and documentation cleanup.
#   2026-06-11, En-Chi Lee (williameclee@gmail.com)
#     - Moved Python backend implementations and auxiliary functions
#       to separate files.
#     - Standardised variable, argument, and function names.
#   2026-07-01, En-Chi Lee (williameclee@gmail.com)
#     - Allowed specifying validity mask in `count_indegree`.
#   2026-07-08, En-Chi Lee (williameclee@gmail.com)
#     - Renamed helper submodule from `aux` to `utils`.
#   2026-07-14, En-Chi Lee (williameclee@gmail.com)
#     - Splitted `geomorphology.flowdir` into submodules.
#   2026-08-03, En-Chi Lee (williameclee@gmail.com)
#     - Implemented functions `find_acyclic_flowdirs` and
#       `find_cyclic_flowdirs` with both FORTRAN and Python
#       backends.


import numpy as np

from formosa.utils import Backend, raise_fortran_error
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

from typing import Optional
import numpy.typing as npt


def _compute_flowdir_simple(
    dem: npt.NDArray[np.number],
    dir_scheme: D8Directions = D8Directions(),
    valids: Optional[npt.NDArray[np.bool_]] = None,
    backend: Backend = "fortran",
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.bool_]]:
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
        `'fortran'` uses the FORTRAN extension for performance,
        while `'python'` uses a pure Python implementation.
        Default backend is `'fortran'`.

    Returns
    -------
    dirs : NDArray[int]
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
    dem: npt.NDArray[np.number],
    dir_scheme: D8Directions = D8Directions(),
    valids: Optional[npt.NDArray[np.bool_]] = None,
    step_size: int = 4,
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.bool_], npt.NDArray[np.integer]]:
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
    dirs : NDArray[int]
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
    dem: npt.NDArray[np.number],
    dir_scheme: D8Directions = D8Directions(),
    valids: Optional[npt.NDArray[np.bool_]] = None,
    fill_depression: bool = False,
    resolve_flat: bool = True,
    step_size: int = 4,
) -> tuple[
    npt.NDArray[np.uint8], npt.NDArray[np.bool_], Optional[npt.NDArray[np.integer]]
]:
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
    dirs: npt.NDArray[np.integer],
    dir_scheme: D8Directions = D8Directions(),
    valids: Optional[npt.NDArray[np.bool_]] = None,
    backend: Backend = "fortran",
) -> npt.NDArray[np.int8]:
    """
    Computes the number of upstream cells (indegree) for each cell in a flow direction grid.

    Parameters
    ----------
    dirs : NDArray[int]
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
        `'fortran'` uses the FORTRAN extension for performance,
        while `'python'` uses a pure Python implementation.
        Default backend is `'fortran'`.

    Returns
    -------
    indegs : NDArray[int8]
        A 2D integer array representing the indegree (number of upstream cells) for each cell.
    """
    if valids is None:
        valids = np.ones(dirs.shape, dtype=bool, order="F")

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
    dirs: npt.NDArray[np.integer],
    indegs: npt.NDArray[np.integer],
    valids: npt.NDArray[np.bool_],
    dir_scheme: D8Directions,
) -> npt.NDArray[np.bool_]:
    """
    Finds acyclic flow cells using the FORTRAN backend.

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
    dirs: npt.NDArray[np.integer],
    dir_scheme: D8Directions = D8Directions(),
    valids: Optional[npt.NDArray[np.bool_]] = None,
    indegs: Optional[npt.NDArray[np.integer]] = None,
    backend: Backend = "fortran",
) -> npt.NDArray[np.bool_]:
    """
    Finds valid cells that do not belong to a directed flow cycle.

    Uses Kahn's algorithm to remove cells reachable from zero-indegree cells.
    Valid cells remaining after the traversal belong to directed cycles.

    Parameters
    ----------
    dirs : NDArray[int]
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
        `'fortran'` uses the FORTRAN extension for performance,
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
        If the FORTRAN backend cannot allocate its workspace.
    RuntimeError
        If the FORTRAN backend reports queue overflow or an unexpected status.
    """
    if dirs.ndim != 2:
        raise ValueError("'dirs' must be a two-dimensional array.")
    if valids is None:
        valids = np.ones(dirs.shape, dtype=bool, order="F")
    elif valids.shape != dirs.shape:
        raise ValueError("Shapes of 'dirs' and 'valids' must match.")

    if indegs is None:
        indegs = count_indegree(
            dirs, dir_scheme=dir_scheme, valids=valids, backend=backend
        )
    elif indegs.shape != dirs.shape:
        raise ValueError("Shapes of 'dirs' and 'indegs' must match.")

    match backend:
        case "python":
            acyclics = flowdir_py.find_acyclic_flowdirs(
                dirs, indegs, valids, dir_scheme=dir_scheme
            )
        case "fortran":
            acyclics = _find_acyclic_flowdirs_fortran(dirs, indegs, valids, dir_scheme)

    return np.asarray(acyclics & valids, dtype=bool, order="F")


def find_cyclic_flowdirs(
    dirs: npt.NDArray[np.integer],
    dir_scheme: D8Directions = D8Directions(),
    valids: Optional[npt.NDArray[np.bool_]] = None,
    indegs: Optional[npt.NDArray[np.integer]] = None,
    backend: Backend = "fortran",
) -> npt.NDArray[np.bool_]:
    """
    Finds valid cells belonging to directed flow cycles.

    Parameters
    ----------
    dirs : NDArray[int]
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
        `'fortran'` uses the FORTRAN extension for performance,
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
        If the FORTRAN backend cannot allocate its workspace.
    RuntimeError
        If the FORTRAN backend reports queue overflow or an unexpected status.
    """
    if valids is None:
        valids = np.ones(dirs.shape, dtype=bool, order="F")

    acyclics = find_acyclic_flowdirs(
        dirs,
        dir_scheme=dir_scheme,
        valids=valids,
        indegs=indegs,
        backend=backend,
    )
    return np.asarray(valids & ~acyclics, dtype=bool, order="F")
