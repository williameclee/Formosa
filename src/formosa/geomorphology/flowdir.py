# Last modified
#   2026-02-11, En-Chi Lee (williameclee@arizona.edu)
#     - Rename flowdir functions to be more descriptive
#   2026-06-09, En-Chi Lee (williameclee@gmail.com)
#     - Added `compute_flow_dist2ridge` function to compute 'distance to ridges'
#     - Added error for missing FORTRAN backend
#     - Removed NumPy type `np.bool` to either `np.bool_` or `bool` for compatibility with newer Numpy versions
#     - Renamed Fortran function call: `compute_masked_flowdir` -> `compute_synthetic_flowdir`
#     - Added `valids` argument to `label_flats` function
#   2026-06-10, En-Chi Lee (williameclee@gmail.com)
#     - Small refactors and documentation cleanup
#   2026-06-11, En-Chi Lee (williameclee@gmail.com)
#     - Moved Python backend implementations and auxiliary functions to separate files
#     - Standardised variable, argument, and function names
#   2026-06-30, En-Chi Lee (williameclee@gmail.com)
#     - Added `x` and `y` into `compute_dist2source` in `compute_dist2ridge`
#     - Changed strahler order output to 8-bit unsigned integer
#     - Added functions `compute_ridgedir` and `compute_ridge_strahler_order`
#   2026-07-01, En-Chi Lee (williameclee@gmail.com)
#     - Opted out of the out-of-bound check in `compute_downstream_indices` in `create_flowgraph`
#     - Allowed specifying validity mask in `count_indegree`
#     - Added function `construct_flowgraph`


import numpy as np

from formosa.geomorphology.d8directions import D8Directions
from .aux import get_neighbour_values, compute_downstream_indices

try:
    from formosa.geomorphology.flowdir_f import flowdir as flowdir_f
except ImportError as err:

    class _MissingFortranBackend:
        def __init__(self, err: ImportError):
            self._err = err

        def __getattr__(self, name):
            raise ImportError(
                "formosa.geomorphology.flowdir_f is required for backend='fortran' but is not available."
            ) from self._err

    flowdir_f = _MissingFortranBackend(err)

import numpy.typing as npt
from typing import Literal, Optional


def fill_depressions(
    dem: npt.NDArray[np.number],
    valids: Optional[npt.NDArray[np.bool_]] = None,
    method: str = "erosion",
) -> npt.NDArray[np.number]:
    assert method in [
        "erosion",
        "dilation",
    ], f"METHOD must be either 'erosion' or 'dilation', got {method} instead"

    from skimage import morphology

    dem_seed = dem.copy()
    if valids is not None:
        if method == "erosion":
            dem[~valids] = np.nanmin(dem[valids])
            seed_value = np.nanmax(dem[valids]) + 1
        else:
            dem[~valids] = np.nanmax(dem[valids])
            seed_value = np.nanmin(dem[valids]) - 1
    else:
        if method == "erosion":
            seed_value = np.nanmax(dem) + 1
        else:
            seed_value = np.nanmin(dem) - 1

    dem_mask = np.full(dem.shape, True, dtype=bool)
    dem_mask[0, :] = False
    dem_mask[-1, :] = False
    dem_mask[:, 0] = False
    dem_mask[:, -1] = False
    dem_seed[dem_mask] = seed_value
    return morphology.reconstruction(dem_seed, dem, method=method).astype(dem.dtype)


def compute_flowdir_simple(
    dem: npt.NDArray[np.number],
    dir_scheme: D8Directions = D8Directions(),
    valids: Optional[npt.NDArray[np.bool_]] = None,
    backend: Literal["fortran", "python"] = "fortran",
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
        The backend to use for computation. 'fortran' uses the Fortran extension for performance, while 'python' uses a pure Python implementation.
        Default is 'fortran'.

    Returns
    -------
    dirs : NDArray[int]
        A 2D integer array representing the flow directions for each cell in the DEM.
    flats : NDArray[bool]
        A boolean mask array where True indicates cells that are part of flat areas.
    """
    match backend:
        case "python":
            from .flowdir_py import _compute_flowdir_simple_py

            dirs, flats = _compute_flowdir_simple_py(dem, dir_scheme=dir_scheme)
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


def find_flat_edges(
    dem: npt.NDArray[np.number],
    dirs: npt.NDArray[np.integer],
    dir_scheme: D8Directions = D8Directions(),
    valids: Optional[npt.NDArray[np.bool_]] = None,
    backend: Literal["fortran", "python"] = "fortran",
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """
    Finds the cells on the edges of flat areas that drain to lower terrain (low edges) and those that are adjacent to higher terrain (high edges).
    From [R. Barnes *et al.* (2014)](https://doi.org/10.1016/j.cageo.2013.01.009), Algorithm 3 (p. 133).

    Parameters
    ----------
    dem : NDArray[number]
        A 2D array representing the digital elevation model (DEM).
    dirs : NDArray[integer]
        A 2D array representing the flow direction for each cell in the DEM.
    dir_scheme : D8Directions, optional
        An instance of D8Directions defining the flow direction scheme that `flowdirs` uses.
        Default is `D8Directions()`.
    valids : NDArray[bool], optional
        A boolean mask array indicating valid cells in the DEM.
        If `None`, all cells are considered valid.
        Default is `None`.
    backend : {'fortran', 'python'}, optional
        The backend to use for computation. 'fortran' uses the Fortran extension for performance, while 'python' uses a pure Python implementation.
        Default is 'fortran'.

    Returns
    -------
    low_edges : NDArray[bool]
        A boolean mask array where True indicates cells that are low edges of flat areas.
    high_edges : NDArray[bool]
        A boolean mask array where True indicates cells that are high edges of flat areas.
    """
    match backend:
        case "python":
            from .flowdir_py import _find_flat_edges_py

            low_edges, high_edges = _find_flat_edges_py(
                dem, dirs, dir_scheme=dir_scheme
            )
        case "fortran":
            if valids is None:
                valids = np.ones(dem.shape, dtype=bool, order="F")

            low_edges, high_edges = flowdir_f.find_flat_edges(
                dem.astype(np.float32, order="F"),
                dirs.astype(np.int32, order="F"),
                valids.astype(bool, order="F"),
                dir_scheme.offsets.astype(np.int32, order="F"),
                dir_scheme.codes.astype(np.uint8, order="F"),
            )

    return (
        low_edges.astype(bool, order="F"),
        high_edges.astype(bool, order="F"),
    )


def label_flats(
    dem: npt.NDArray[np.number],
    seeds: npt.NDArray[np.bool_],
    valids: Optional[npt.NDArray[np.bool_]] = None,
    dir_scheme: D8Directions = D8Directions(),
) -> npt.NDArray[np.int32]:
    """
    Separates and labels inidividual flat areas in a DEM.
    From [R. Barnes *et al.* (2014)](https://doi.org/10.1016/j.cageo.2013.01.009), Algorithm 4 (p. 133).

    Parameters
    ----------
    dem : NDArray[number]
        A 2D array representing the digital elevation model (DEM).
    seeds : NDArray[bool]
        Either a boolean mask array indicating flat area locations, or a 2D integer array of coordinates, or an iterable of coordinate pairs.
    valids : NDArray[bool], optional
        A boolean mask array indicating valid cells in the DEM.
        If `None`, all cells are considered valid.
        Default is `None`.
    dir_scheme : D8Directions, optional
        An instance of `D8Directions` defining the flow direction scheme.
        Default is `D8Directions()`.

    Returns
    -------
    labels : NDArray[int]
        A 2D integer array where each flat region is labeled with a unique integer.

    Raises
    ------
    TypeError
        If the input seeds is not of the expected type or format.
    ValueError
        If the shapes of the input arrays do not match the expected dimensions.
    """
    assert (
        dem.shape == seeds.shape
    ), f"Shapes for dem ({dem.shape}) and seeds ({seeds.shape}) do not match."
    if valids is not None:
        assert (
            dem.shape == valids.shape
        ), f"Shapes for dem ({dem.shape}) and valids ({valids.shape}) do not match."
    else:
        valids = np.ones(dem.shape, dtype=bool, order="F")

    labels = flowdir_f.label_flats(
        dem.astype(np.float32, order="F"),
        seeds.astype(bool, order="F"),
        valids.astype(bool, order="F"),
        dir_scheme.offsets.astype(np.int32, order="F"),
    )

    return labels.astype(np.int32, order="F")


def find_ambiguous(
    dem: npt.NDArray[np.number],
    dir_scheme: D8Directions = D8Directions(),
) -> npt.NDArray[np.bool_]:
    """
    Detects ambiguous flow directions in a DEM, where multiple neighbouring cells have the same minimum elevation.

    Parameters
    ----------
    dem : NDArray[number]
        A 2D array representing the digital elevation model (DEM).
    dir_scheme : D8Directions, optional
        An instance of `D8Directions` defining the flow direction scheme.
        Default is `D8Directions()`.

    Returns
    -------
    ambiguities : NDArray[bool]
        A boolean mask array where True indicates cells with ambiguous flow directions.
    """
    neighbours, _, _ = get_neighbour_values(dem, dir_scheme=dir_scheme)
    min_neighbours = np.min(neighbours, axis=0)
    ambiguities = np.sum(neighbours == min_neighbours, axis=0) > 1
    ambiguities = ambiguities & ~(find_flat(dem))
    return ambiguities


def find_flat(
    dem: npt.NDArray[np.number],
    valids: Optional[npt.NDArray[np.bool_]] = None,
    only_min: bool = True,
    dir_scheme: D8Directions = D8Directions(window=3),
) -> npt.NDArray[np.bool_]:
    """
    Identifies flat areas in a DEM where cells have no lower neighbouring cells.

    Parameters
    ----------
    dem : NDArray[number]
        A 2D array representing the digital elevation model (DEM).
    valid : NDArray[bool], optional
        A boolean mask array indicating valid cells in the DEM.
        If `None`, all cells are considered valid.
        Default is `None`.
    only_min : bool, optional
        If True, only cells that are equal to the minimum of their neighbours are considered flat.
        If False, cells equal to any neighbour are considered flat.
        Default is True.
    dir_scheme : D8Directions, optional
        An instance of D8Directions defining the neighbour offsets.
        Default is D8Directions(window=3).

    Returns
    -------
    flats : NDArray[bool]
        A boolean mask array where True indicates cells that are part of flat areas.
    """
    if valids is not None and np.any(~valids):
        dem[~valids] = np.max(dem[~valids]) + 1

    neighbours, _, _ = get_neighbour_values(
        dem, dir_scheme=dir_scheme, pad_value=np.nan, include_self=False
    )
    if only_min:
        flats = dem == np.nanmin(neighbours, axis=0)
    else:
        flats = np.any(dem == neighbours, axis=0)

    if valids is not None:
        flats = flats & valids
    return flats


def create_pushing_syn_grad(
    labels: npt.NDArray[np.number],
    high_edges: npt.NDArray[np.bool_],
    dir_scheme: D8Directions = D8Directions(),
) -> npt.NDArray[np.int32]:
    """
    Produces a synthetic elevation that decreases away from 'high edges' of flats.
    Modified from [R. Barnes *et al.* (2014)](https://doi.org/10.1016/j.cageo.2013.01.009), Algorithm 5 (p. 133–134).

    Parameters
    ----------
    labels : NDArray[number]
        A 2D array where each flat region is labeled with a unique integer.
        It is assumed that non-flat areas are labeled with 0, and flat areas have positive integer labels starting from 1 (the Fortran extension relies on this).
    high_edges : NDArray[bool]
        A boolean mask array indicating high edge locations.
    dir_scheme : D8Directions, optional
        An instance of D8Directions defining the flow direction scheme, here it is used to determine the offsets for neighbor cells.
        Default is `D8Directions()`.

    Returns
    -------
    z_syn : NDArray[int32]
        A 2D integer array representing the synthetic elevation that increases away from high edges within each flat region.

    Raises
    ------
    TypeError
        If the input high_edges is not of the expected type or format.
    ValueError
        If the shapes of the input arrays do not match the expected dimensions.
    """
    assert (
        labels.shape == high_edges.shape
    ), f"Shapes for labels ({labels.shape}) and high_edges ({high_edges.shape}) do not match."

    z_syn = flowdir_f.create_pushing_syn_grad(
        labels.astype(np.int32, order="F"),
        high_edges.astype(bool, order="F"),
        dir_scheme.offsets.astype(np.int32, order="F"),
    )
    return z_syn.astype(np.int32, order="F")


def create_pulling_syn_grad(
    labels: npt.NDArray[np.number],
    low_edges: npt.NDArray[np.bool_],
    dir_scheme: D8Directions = D8Directions(),
) -> npt.NDArray[np.integer]:
    """
    Produces a synthetic elevation that drains towards 'low edges' of flats.
    Modified from [R. Barnes *et al.* (2014)](https://doi.org/10.1016/j.cageo.2013.01.009), Algorithm 6 (p. 134).

    Parameters
    ----------
    labels : NDArray[number]
        A 2D array where each flat region is labeled with a unique integer.
        It is assumed that non-flat areas are labeled with 0, and flat areas have positive integer labels starting from 1 (the Fortran extension relies on this).
    low_edges : NDArray[bool]
        A boolean mask array indicating low edge locations.
    dir_scheme : D8Directions, optional
        An instance of D8Directions defining the flow direction scheme, here it is used to determine the offsets for neighbor cells.
        Default is `D8Directions()`.

    Returns
    -------
    z_syn : NDArray[integer]
        A 2D integer array representing the synthetic elevation that increases towards low edges within each flat region.

    Raises
    ------
    TypeError
        If the input low_edges is not of the expected type or format.
    ValueError
        If the shapes of the input arrays do not match the expected dimensions.
    """
    z_syn = flowdir_f.create_pulling_syn_grad(
        labels.astype(np.int32, order="F"),
        low_edges.astype(bool, order="F"),
        dir_scheme.offsets.astype(np.int32, order="F"),
    )
    return z_syn


def compute_syn_flowdir(
    z: npt.NDArray[np.integer | np.floating],
    labels: npt.NDArray[np.integer],
    dir_scheme: D8Directions = D8Directions(),
    backend: Literal["fortran", "python"] = "fortran",
) -> npt.NDArray[np.uint8]:
    """
    Computes flow directions within flat areas using synthetic elevation.
    Very similar to the naive flow direction computation, but only search within the same flat area.

    Parameters
    ----------
    z : NDArray[int | float]
        A 2D array representing the synthetic elevation within flat areas.
    labels : NDArray[int]
        A 2D array where each flat region is labeled with a unique integer.
    dir_scheme : D8Directions, optional
        An instance of `D8Directions` defining the flow direction scheme.
        Default is `D8Directions()`.
    backend : {'fortran', 'python'}, optional
        The backend to use for computation. 'fortran' uses the Fortran extension for performance,
        while 'python' uses a pure Python implementation. Default is 'fortran'.

    Returns
    -------
    dirs : NDArray[int]
        A 2D integer array representing the flow directions within flat areas.
    """
    match backend:
        case "python":
            from .flowdir_py import _compute_masked_flowdir_py

            dirs = _compute_masked_flowdir_py(z, labels, dir_scheme=dir_scheme)
        case "fortran":
            dirs = flowdir_f.compute_syn_flowdir(
                z.astype(np.int32, order="F"),
                labels.astype(np.int32, order="F"),
                dir_scheme.offsets.astype(np.int32, order="F"),
                dir_scheme.codes.astype(np.uint8, order="F"),
            )

    return dirs.astype(np.uint8, order="F")


def compute_flowdir_complete(
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

    dirs, flats = compute_flowdir_simple(dem, dir_scheme=dir_scheme, valids=valids)
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
    fill_depression_method: str = "erosion",
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
        A 2D array representing the digital elevation model (DEM).
    dir_scheme : D8Directions, optional
        An instance of `D8Directions` defining the flow direction scheme.
        Default is `D8Directions()`.
    valids : NDArray[bool], optional
        A boolean mask array indicating valid cells in the DEM.
        If `None`, all cells are considered valid.
        Default is `None`.
    fill_depression : bool, optional
        Whether to fill depressions in the DEM before computing flow directions.
        Default is False.
    fill_depression_method : {'erosion', 'dilation'}, optional
        The method to use for filling depressions, either 'erosion' or 'dilation'.
        Default is 'erosion'.
    resolve_flat : bool, optional
        Whether to resolve flat areas using synthetic elevations.
        Default is True.
    step_size : int, optional
        The increment in synthetic elevation per step away from low edges to avoid ties when combining synthetic elevations.
        Default is 4.

    Returns
    -------
    dirs : NDArray[uint8]
        A 2D integer array representing the flow directions for each cell in the DEM.
    flats : NDArray[bool]
        A boolean mask array where True indicates cells that are part of flat areas.
    syn_grads : NDArray[int] | None
        A 2D integer array representing the synthetic elevation that resolves flat areas, or None if resolve_flat is False.
    """
    if fill_depression:
        dem = fill_depressions(dem, valids=valids, method=fill_depression_method)
    if resolve_flat:
        dirs, flats, syn_grads = compute_flowdir_complete(
            dem, dir_scheme=dir_scheme, valids=valids, step_size=step_size
        )
    else:
        dirs, flats = compute_flowdir_simple(dem, dir_scheme=dir_scheme, valids=valids)
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
    backend: Literal["fortran", "python"] = "fortran",
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
        The backend to use for computation. 'fortran' uses the Fortran extension for performance, while 'python' uses a pure Python implementation.
        Note: the Python implementation is unmaintained.
        Default is 'fortran'.

    Returns
    -------
    indegs : NDArray[int8]
        A 2D integer array representing the indegree (number of upstream cells) for each cell.
    """
    if valids is None:
        valids = np.ones(dirs.shape, dtype=bool, order="F")

    match backend:
        case "python":
            from .flowdir_py import _count_indegree_py

            indegs = _count_indegree_py(dirs, dir_scheme=dir_scheme, valids=valids)
        case "fortran":
            indegs = flowdir_f.count_indegree(
                dirs.astype(np.uint8, order="F"),
                valids.astype(bool, order="F"),
                dir_scheme.offsets.astype(np.int32, order="F"),
                dir_scheme.codes.astype(np.uint8, order="F"),
            )

    return indegs.astype(np.int8, order="F")


def create_flowgraph(
    dirs: npt.NDArray[np.integer],
    valids: Optional[npt.NDArray[np.bool_]] = None,
    dir_scheme: D8Directions = D8Directions(),
    x: Optional[npt.NDArray[np.number]] = None,
    y: Optional[npt.NDArray[np.number]] = None,
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]]:
    """
    Computes a graph representation of the flow directions in a flow direction grid.

    Parameters
    ----------
    dirs : NDArray[int]
        A 2D array representing the flow directions for each cell.
    valids : NDArray[bool], optional
        A boolean mask array indicating valid cells in the flow direction grid.
        If `None`, all cells are considered valid.
        Default is `None`.
    directions : D8Directions, optional
        An instance of `D8Directions` defining the flow direction scheme.
        Default is `D8Directions()`.
    x : NDArray[number], optional
        A 2D array representing the x-coordinates of each cell.
        If provided, the graph will use these coordinates instead of grid indices.
        Default is `None`.
    y : NDArray[number], optional
        A 2D array representing the y-coordinates of each cell.
        If provided, the graph will use these coordinates instead of grid indices.
        Default is `None`.

    Returns
    -------
    graphi : NDArray[int]
        A 1D array representing the row indices of the graph edges.
    graphj : NDArray[int]
        A 1D array representing the column indices of the graph edges.
    """
    if valids is not None:
        assert (
            valids.shape == dirs.shape
        ), f"Shape for dlowdirs and valids mask must match, but got valid shape {dirs.shape} and flowdirs shape {valids.shape} instead"
    else:
        valids = np.full(dirs.shape, True, dtype=bool)

    i, j = np.meshgrid(
        np.arange(dirs.shape[0], dtype=np.int32),
        np.arange(dirs.shape[1], dtype=np.int32),
        indexing="ij",
    )
    dsi, dsj, _, ds_valids = compute_downstream_indices(
        dirs, dir_scheme=dir_scheme, check=False
    )

    if x is not None and y is not None:
        j, i = x, y

        # Map i,j to actual coordinates
        dsx = np.full_like(dsj, np.nan, dtype=np.float64)
        dsy = np.full_like(dsj, np.nan, dtype=np.float64)
        dsx[ds_valids] = x[dsi[ds_valids], dsj[ds_valids]]
        dsy[ds_valids] = y[dsi[ds_valids], dsj[ds_valids]]
        dsi, dsj = dsy, dsx

    graphi = np.stack(
        (
            i[valids & ds_valids],
            dsi[valids & ds_valids],
            np.full(i[valids & ds_valids].size, np.nan),
        ),
        axis=1,
    ).ravel(order="C")
    graphj = np.stack(
        (
            j[valids & ds_valids],
            dsj[valids & ds_valids],
            np.full(j[valids & ds_valids].size, np.nan),
        ),
        axis=1,
    ).ravel(order="C")
    return graphi, graphj


def compute_flow_accumulation(
    dirs: npt.NDArray[np.integer],
    valids: Optional[npt.NDArray[np.bool_]] = None,
    weights: Optional[npt.NDArray[np.floating]] = None,
    indegs: Optional[npt.NDArray[np.integer]] = None,
    dsij: Optional[npt.NDArray[np.integer]] = None,
    dir_scheme: D8Directions = D8Directions(),
    backend: Literal["fortran", "python"] = "fortran",
) -> npt.NDArray[np.float32]:
    """
    Computes flow accumulation for each cell in a flow direction grid.

    Parameters
    ----------
    dirs : NDArray[int]
        A 2D array representing the flow directions for each cell.
    valids : NDArray[bool], optional
        A boolean mask array indicating valid cells in the flow direction grid.
        If `None`, all cells are considered valid.
        Default is `None`.
    weights : NDArray[float], optional
        A 2D array of weights for each cell, representing the contribution of each cell to its downstream cell.
        If `None`, each valid cell contributes a weight of 1.0.
        Default is `None`.
    indegs : NDArray[int], optional
        A 2D array representing the indegree (number of upstream cells) for each cell.
        If `None`, `indegs` are computed from the flow direction grid.
        Default is `None`.
    dsij : NDArray[int], optional
        A 2D array of downstream cell indices for each cell.
        If `None`, downstream indices are computed from the flow direction grid.
        Default is `None`.
    dir_scheme : D8Directions, optional
        An instance of `D8Directions` defining the flow direction scheme.
        Default is `D8Directions()`.
    backend : {'fortran', 'python'}, optional
        The backend to use for computation. 'fortran' uses the Fortran extension for performance, while 'python' uses a pure Python implementation.
        Default is 'fortran'.

    Returns
    -------
    accums : NDArray[float32]
        A 2D array representing the flow accumulation for each cell.
    """
    match backend:
        case "python":
            from .flowdir_py import _compute_flow_accumulation_py

            accums = _compute_flow_accumulation_py(
                dirs,
                valids=valids,
                weights=weights,
                indegs=indegs,
                dsij=dsij,
                dir_scheme=dir_scheme,
            )
        case "fortran":
            if indegs is None:
                indegs = count_indegree(dirs, dir_scheme=dir_scheme)

            if valids is None:
                valids = np.ones(dirs.shape, dtype=bool)

            if weights is None:
                weights = np.where(valids, 1.0, 0.0).astype(np.float32)

            accums = flowdir_f.compute_flow_accumulation(
                dirs.astype(np.uint8, order="F"),
                valids.astype(bool, order="F"),
                weights.astype(np.float32, order="F"),
                indegs.astype(np.int8, order="F"),
                dir_scheme.offsets.astype(np.int32, order="F"),
                dir_scheme.codes.astype(np.uint8, order="F"),
            )

    return accums.astype(np.float32, order="F")


def compute_flow_strahler_order(
    dirs: npt.NDArray[np.integer],
    dir_scheme: D8Directions = D8Directions(),
    valids: Optional[npt.NDArray[np.bool_]] = None,
    indegs: Optional[npt.NDArray[np.integer]] = None,
    backend: Literal["fortran", "python"] = "fortran",
) -> npt.NDArray[np.uint8]:
    """
    Computes the Strahler order for each cell in a flow direction grid.

    Parameters
    ----------
    dirs : NDArray[int], optional
        2D array representing the flow directions for each cell.
    dir_scheme : D8Directions, optional
        Instance of `D8Directions` defining the flow direction scheme.
        Default is `D8Directions()`.
    valids : NDArray[bool], optional
        Boolean mask array indicating valid cells in the flow direction grid.
        If `None`, all cells are considered valid.
        Default is `None`.
    indegs : NDArray[int], optional
        2D array representing the number of upstream cells for each cell.
        If `None`, it will be computed from the flow direction grid.
        Default is `None`.
    backend : {'fortran', 'python'}, optional
        Backend to use for computation
        'fortran' uses the Fortran extension for performance, while 'python' uses a pure Python implementation.
        Default is 'fortran'.

    Returns
    -------
    orders : NDArray[uint8]
        2D integer array representing the Strahler order for each cell
    """
    match backend:
        case "python":
            from .flowdir_py import _compute_flow_strahler_order_py

            orders = _compute_flow_strahler_order_py(
                dirs=dirs,
                dir_scheme=dir_scheme,
                indegs=indegs,
            )
        case "fortran":
            if valids is None:
                valids = np.ones(dirs.shape, dtype=bool)

            if indegs is None:
                indegs = count_indegree(dirs, dir_scheme=dir_scheme, backend="fortran")

            orders = flowdir_f.compute_flow_strahler_order(
                dirs.astype(np.uint8, order="F"),
                valids.astype(bool, order="F"),
                indegs.astype(np.int8, order="F"),
                dir_scheme.offsets.astype(np.int32, order="F"),
                dir_scheme.codes.astype(np.uint8, order="F"),
            )
            orders[~valids] = 0
    return orders.astype(np.uint8, order="F")


def construct_flowgraph(
    dirs: npt.NDArray[np.integer],
    dir_scheme: D8Directions = D8Directions(),
    valids: Optional[npt.NDArray[np.bool_]] = None,
    min_order: int = 2,
    orders: Optional[npt.NDArray[np.integer]] = None,
    preserve_junctions: bool = True,
    backend: Literal["fortran", "python"] = "fortran",
) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """
    Constructs a flow graph from a flow direction grid.

    Parameters
    ----------
    dirs : NDArray[int], optional
        A 2D array representing the flow directions for each cell
    dir_scheme : D8Directions, optional
        Instance of `D8Directions` defining the flow direction scheme
        Default is `D8Directions()`.
    valids : NDArray[bool], optional
        Boolean mask array indicating valid cells in the flow direction grid
        If `None`, all cells are considered valid.
        Default is `None`.
    min_order : int, optional
        Minimum Strahler order to include in the flow graph (see `orders`)
        Default is 2.
    orders : NDArray[uint8], optional
        2D integer array representing the Strahler order for each cell
        If `None`, it will be computed from the flow direction grid.
        Default is `None`.
    preserve_junctions : bool, optional
        Whether to preserve junctions in the flow graph
        Default is True.
    backend : {'fortran', 'python'}, optional
        The backend to use for computation
        'fortran' uses the Fortran extension for performance, while 'python' uses a pure Python implementation.
        Default is 'fortran'.

    Returns
    -------
    arc_orders : NDArray[int8]
        1D array representing the Strahler order for each arc in the flow graph
    vertex_ijs : NDArray[int8]
        V-by-2 array containing the ordered (i, j) incices of all arcs, concactinated together
    vertex_startends : NDArray[int32]
        A-by-2 array containing the indices of where each arc starts and ends in `vertex_ijs`
    """
    if valids is None:
        valids = np.ones(dirs.shape, dtype=bool)
    if orders is None:
        orders = compute_flow_strahler_order(
            dirs,
            dir_scheme=dir_scheme,
            backend=backend,
        )

    # Find seed cells to start with
    valids = valids & (orders >= min_order)
    ncells = np.sum(valids)
    indegs = count_indegree(dirs, dir_scheme=dir_scheme, valids=valids, backend=backend)
    seeds = valids & (indegs == 0)

    match backend:
        case "python":
            from .flowdir_py import _construct_flowgraph_py

            narcs, nvertices, arc_orders, vertex_ijs, vertex_startends = (
                _construct_flowgraph_py(
                    dirs=dirs,
                    dir_scheme=dir_scheme,
                    valids=valids,
                    orders=orders,
                    indegs=indegs,
                    seeds=seeds,
                    preserve_junctions=preserve_junctions,
                    ncells=ncells,
                )
            )
        case "fortran":
            narcs, nvertices, arc_orders, vertex_ijs, vertex_startends = (
                flowdir_f.construct_flowgraph(
                    dirs.astype(np.uint8, order="F"),
                    valids.astype(bool, order="F"),
                    orders.astype(np.int16, order="F"),
                    seeds.astype(np.bool_, order="F"),
                    indegs.astype(np.int8, order="F"),
                    dir_scheme.offsets.astype(np.int32, order="F"),
                    dir_scheme.codes.astype(np.uint8, order="F"),
                    preserve_junctions,
                    ncells,
                )
            )
            # Convert from 1-based index to 0-based index
            vertex_ijs -= 1
            vertex_startends -= 1

    arc_orders = arc_orders[:narcs].T.copy(order="C")
    vertex_startends = vertex_startends[:, :narcs].T.copy(order="C")
    vertex_ijs = vertex_ijs[:, :nvertices].T.copy(order="C")

    return arc_orders, vertex_ijs, vertex_startends


def compute_dist2source(
    dirs: npt.NDArray[np.integer],
    dir_scheme: D8Directions = D8Directions(),
    x: Optional[npt.NDArray[np.number]] = None,
    y: Optional[npt.NDArray[np.number]] = None,
    valids: Optional[npt.NDArray[np.bool_]] = None,
    indegs: Optional[npt.NDArray[np.integer]] = None,
) -> npt.NDArray[np.float32]:
    """
    Computes the distance downstream along flow directions for each cell in the flow direction grid.

    Parameters
    ----------
    dirs : NDArray[int]
        A 2D array representing the flow direction for each cell.
    dir_scheme : D8Directions, optional
        An instance of `D8Directions` defining the flow direction scheme.
        Default is `D8Directions()`.
    x : NDArray[int | float], optional
        A 2D array representing the x-coordinates of each cell. If `None`, cell indices are used.
        Default is `None`.
    y : NDArray[int | float], optional
        A 2D array representing the y-coordinates of each cell. If `None`, cell indices are used.
        Default is `None`.
    valids : NDArray[bool], optional
        A boolean mask array indicating valid cells in the flow direction grid.
        If `None`, all cells are considered valid.
        Default is `None`.
    indegs : NDArray[int], optional
        A 2D array representing the indegree (number of upstream cells) for each cell.
        If `None`, indegs are computed from the flow direction grid.
        Default is `None`.

    Returns
    -------
    dists : NDArray[float32]
        A 2D array representing the downstream distance for each cell.

    Raises
    ------
    TypeError
        If the input arrays are not of the expected type or format.
    ValueError
        If the shapes of the input arrays do not match the expected dimensions.
    """
    if valids is None:
        valids = np.ones(dirs.shape, dtype=bool)
    elif isinstance(valids, np.ndarray):
        assert (
            valids.shape == dirs.shape
        ), f"Shape for flow direction ({dirs.shape}) and valid mask ({valids.shape}) do not match."
    else:
        raise TypeError(f"Valid mask must be a NumPy array (got {type(valids)}).")
    if x is not None and y is not None:
        assert (
            x.shape == dirs.shape and y.shape == dirs.shape
        ), f"Shapes for flow direction ({dirs.shape}) and x ({x.shape}) and y ({y.shape}) must match."
    else:
        x = np.arange(dirs.shape[1], dtype=np.float32)
        y = np.arange(dirs.shape[0], dtype=np.float32)
        x, y = np.meshgrid(x, y, indexing="xy")
    if indegs is None:
        indegs = count_indegree(dirs, dir_scheme=dir_scheme)
    elif isinstance(indegs, np.ndarray):
        assert (
            indegs.shape == dirs.shape
        ), f"Shape for flow direction ({dirs.shape}) and indegree ({indegs.shape}) do not match."
    else:
        raise TypeError(f"Indegree must be a NumPy array (got {type(indegs)}).")

    dists = flowdir_f.compute_dist2source(
        dirs.astype(np.uint8, order="F"),
        valids.astype(bool, order="F"),
        x.astype(np.float32, order="F"),
        y.astype(np.float32, order="F"),
        indegs.astype(np.int8, order="F"),
        dir_scheme.offsets.astype(np.int32, order="F"),
        dir_scheme.codes.astype(np.uint8, order="F"),
    )
    return dists.astype(np.float32, order="F")


def label_watersheds(
    dirs: npt.NDArray[np.integer],
    dir_scheme: D8Directions = D8Directions(),
    valids: Optional[npt.NDArray[np.bool_]] = None,
    backend: Literal["fortran", "python"] = "fortran",
) -> npt.NDArray[np.int32]:
    """
    Finds and labels watersheds in a DEM based on flow direction.

    Parameters
    ----------
    dirs : NDArray[int]
        A 2D array representing the flow direction for each cell.
    dir_scheme : D8Directions, optional
        An instance of `D8Directions` defining the flow direction scheme.
        Default is `D8Directions()`.
    valids : NDArray[bool], optional
        A boolean mask array indicating valid cells in the flow direction grid.
        If `None`, all non-NaN cells in flowdirs are considered valid.
        Default is `None`.

    Returns
    -------
    watersheds : NDArray[int32]
        A 2D array where each watershed is labeled with a unique integer.
    """
    match backend:
        case "python":
            from .flowdir_py import _label_watersheds_py

            watersheds = _label_watersheds_py(
                dirs=dirs,
                dir_scheme=dir_scheme,
                valids=valids,
            )
        case "fortran":
            if valids is None:
                valids = np.ones(dirs.shape, dtype=bool)
            elif isinstance(valids, np.ndarray):
                assert (
                    valids.shape == dirs.shape
                ), f"Shape for flow direction ({dirs.shape}) and valid mask ({valids.shape}) do not match."
                valids = valids.astype(bool, copy=False) & (~np.isnan(dirs))
                dirs = np.where(valids, dirs, np.nan)
            else:
                raise TypeError(
                    f"Valid mask must be a NumPy array (got {type(valids)})."
                )

            watersheds = flowdir_f.label_watersheds(
                dirs.astype(np.uint8, order="F"),
                valids.astype(bool, order="F"),
                dir_scheme.offsets.astype(np.int32, order="F"),
                dir_scheme.codes.astype(np.uint8, order="F"),
            )
    return watersheds.astype(np.int32, order="F")


def compute_dist2sink(
    dirs: npt.NDArray[np.integer],
    dir_scheme: D8Directions = D8Directions(),
    x: Optional[npt.NDArray[np.number]] = None,
    y: Optional[npt.NDArray[np.number]] = None,
    valids: Optional[npt.NDArray[np.bool_]] = None,
) -> npt.NDArray[np.float32]:
    """
    Computes the distance upstream along flow directions for each cell in the flow direction grid.

    Parameters
    ----------
    dirs : NDArray[int]
        A 2D array representing the flow direction for each cell.
    dir_scheme : D8Directions, optional
        An instance of `D8Directions` defining the flow direction scheme.
        Default is `D8Directions()`.
    x : NDArray[int | float], optional
        A 2D array representing the x-coordinates of each cell. If `None`, a default grid will be created.
    y : NDArray[int | float], optional
        A 2D array representing the y-coordinates of each cell. If `None`, a default grid will be created.
    valids : NDArray[bool], optional
        A boolean mask array where `True` indicates valid cells. If `None`, all non-NaN cells in `dirs` are considered valid.

    Returns
    -------
    dists : NDArray[float32]
        A 2D array representing the upstream distance for each cell.
    """
    if valids is None:
        valids = ~np.isnan(dirs)
    elif isinstance(valids, np.ndarray):
        assert (
            valids.shape == dirs.shape
        ), f"Shape for flow direction ({valids.shape}) and valid mask ({dirs.shape}) do not match."
        valids = valids.astype(bool, copy=False) & (~np.isnan(dirs))
        dirs = np.where(valids, dirs, np.nan)
    else:
        raise TypeError(
            f"Validity mask must be either None or a numpy array, (got {type(valids)})."
        )
    if x is not None and y is not None:
        assert (
            x.shape == dirs.shape and y.shape == dirs.shape
        ), f"Shapes for flow direction ({dirs.shape}) and x ({x.shape}) and y ({y.shape}) must match."
    else:
        x = np.arange(dirs.shape[1], dtype=np.float32)
        y = np.arange(dirs.shape[0], dtype=np.float32)
        x, y = np.meshgrid(x, y, indexing="xy")

    dists = flowdir_f.compute_dist2sink(
        dirs.astype(np.uint8, order="F"),
        x.astype(np.float32, order="F"),
        y.astype(np.float32, order="F"),
        valids.astype(bool, order="F"),
        dir_scheme.offsets.astype(np.int32, order="F"),
        dir_scheme.codes.astype(np.uint8, order="F"),
    )
    return dists.astype(np.float32, order="F")


def compute_dist2conf_max(
    dirs: npt.NDArray[np.integer],
    valids: Optional[npt.NDArray[np.bool_]] = None,
    x: Optional[npt.NDArray[np.number]] = None,
    y: Optional[npt.NDArray[np.number]] = None,
    watershed_labels: Optional[npt.NDArray[np.integer]] = None,
    dir_scheme: D8Directions = D8Directions(),
) -> npt.NDArray[np.float32]:
    """
    Computes the maximum distance to confluence for each cell with its neighbours in the flow direction grid.
    If the cell does not share a confluence with any of its neighbours, the distance to sink is returned instead.
    This field can be used as an proxy for the ridge network, where cells with a larger distance to confluence are more likely to be part of the ridge network.
    See `compute_flow_dist2ridge` for computing the distance to ridge based on this field.

    Parameters
    ----------
    dirs : NDArray[uint8]
        A 2D array representing the flow directions for each cell.
    valids : NDArray[bool], optional
        A boolean mask array where `True` indicates valid cells. If `None`, all cells are considered valid.
        Default is `None`.
    x : NDArray[int | float], optional
        A 2D array representing the x-coordinates of each cell. If `None`, a default grid will be created.
        Default is `None`.
    y : NDArray[int | float], optional
        A 2D array representing the y-coordinates of each cell. If `None`, a default grid will be created.
        Default is `None`.
    watershed_labels : NDArray[int], optional
        A 2D array representing labels for different watersheds in the flow direction grid. Since celss in different watersheds do not share confluences, providing watershed labels can skip unnecessary comparisons.
        If `None`, all cells are assigned the same label.
        Default is `None`.
    dir_scheme : D8Directions, optional
        An instance of `D8Directions` defining the flow direction scheme.
        Default is `D8Directions()`.

    Returns
    -------
    NDArray[float32]
        A 2D array representing the maximum distance to confluence for each cell.
    """
    if valids is None:
        valids = np.ones(dirs.shape, dtype=bool)
    elif isinstance(valids, np.ndarray):
        assert (
            valids.shape == dirs.shape
        ), f"Shape for flow direction ({dirs.shape}) and valid mask ({valids.shape}) do not match."
    else:
        raise TypeError(f"Valid mask must be a NumPy array (got {type(valids)}).")
    if x is not None and y is not None:
        assert (
            x.shape == dirs.shape and y.shape == dirs.shape
        ), f"Shapes for flow direction ({dirs.shape}) and x ({x.shape}) and y ({y.shape}) must match."
    else:
        x = np.arange(dirs.shape[1], dtype=np.float32)
        y = np.arange(dirs.shape[0], dtype=np.float32)
        x, y = np.meshgrid(x, y, indexing="xy")
    if watershed_labels is None:
        watershed_labels = np.ones(dirs.shape, dtype=np.int32)
    elif isinstance(watershed_labels, np.ndarray):
        assert (
            watershed_labels.shape == dirs.shape
        ), f"Shape for flow direction ({dirs.shape}) and labels ({watershed_labels.shape}) do not match."
    else:
        raise TypeError(f"Labels must be a NumPy array (got {type(watershed_labels)}).")

    bmax = flowdir_f.compute_max_branch_dist(
        dirs.astype(np.uint8, order="F"),
        valids.astype(bool, order="F"),
        x.astype(np.float32, order="F"),
        y.astype(np.float32, order="F"),
        watershed_labels.astype(np.int32, order="F"),
        dir_scheme.offsets.astype(np.int32, order="F"),
        dir_scheme.codes.astype(np.uint8, order="F"),
    )
    return bmax.astype(np.float32, order="F")


def compute_ridgedir(
    dirs: npt.NDArray[np.integer],
    dir_scheme: D8Directions = D8Directions(),
    valids: Optional[npt.NDArray[np.bool_]] = None,
    x: Optional[npt.NDArray[np.number]] = None,
    y: Optional[npt.NDArray[np.number]] = None,
    watershed_labels: Optional[npt.NDArray[np.integer]] = None,
) -> npt.NDArray[np.uint8]:
    bmax = compute_dist2conf_max(
        dirs,
        valids=valids,
        x=x,
        y=y,
        watershed_labels=watershed_labels,
        dir_scheme=dir_scheme,
    )
    bmaxdirs, _, _ = compute_flowdir(
        -bmax, dir_scheme=dir_scheme, valids=valids, fill_depression=True
    )
    return bmaxdirs.astype(np.uint8, order="F")


def compute_dist2ridge(
    dirs: npt.NDArray[np.integer],
    dir_scheme: D8Directions = D8Directions(),
    valids: Optional[npt.NDArray[np.bool_]] = None,
    x: Optional[npt.NDArray[np.number]] = None,
    y: Optional[npt.NDArray[np.number]] = None,
    watershed_labels: Optional[npt.NDArray[np.integer]] = None,
    dir_is_ridge: bool = False,
) -> npt.NDArray[np.float32]:
    """
    Computes the 'distance to ridge' for each cell in the flow direction grid.
    The ridge network/intensity is defined as the maximum distance to confluence (see `compute_flow_dist2conf_max`), and the distance to ridge is computed as the downstream distance traversing the inverse of the intensity.

    Parameters
    ----------
    dirs : NDArray[int]
        A 2D array representing the flow directions for each cell.
    dir_scheme : D8Directions, optional
        An instance of `D8Directions` defining the flow direction scheme.
        Default is `D8Directions()`.
    valids : NDArray[bool], optional
        A boolean mask array where `True` indicates valid cells. If `None`, all cells are considered valid.
        Default is `None`.
    x : NDArray[int | float], optional
        A 2D array representing the x-coordinates of each cell. If `None`, a default grid will be created.
        Default is `None`.
    y : NDArray[int | float], optional
        A 2D array representing the y-coordinates of each cell. If `None`, a default grid will be created.
        Default is `None`.
    watershed_labels : NDArray[int], optional
        A 2D array representing labels for different watersheds in the flow direction grid. Since celss in different watersheds do not share confluences, providing watershed labels can skip unnecessary comparisons.
        If `None`, all cells are assigned the same label.
        Default is `None`.

    Returns
    -------
    bmaxdists : NDArray[float32]
        A 2D array representing the distance to ridge for each cell.
    """
    if dir_is_ridge:
        bmaxdirs = dirs
    else:
        bmaxdirs = compute_ridgedir(
            dirs,
            dir_scheme=dir_scheme,
            valids=valids,
            x=x,
            y=y,
            watershed_labels=watershed_labels,
        )
    bmaxdists = compute_dist2source(
        bmaxdirs, dir_scheme=dir_scheme, x=x, y=y, valids=valids
    )
    return bmaxdists


def compute_ridge_strahler_order(
    dirs: npt.NDArray[np.integer],
    dir_scheme: D8Directions = D8Directions(),
    valids: Optional[npt.NDArray[np.bool_]] = None,
    indegs: Optional[npt.NDArray[np.integer]] = None,
    backend: Literal["fortran", "python"] = "fortran",
    dir_is_ridge: bool = False,
) -> npt.NDArray[np.uint8]:
    if dir_is_ridge:
        bmaxdirs = dirs
    else:
        bmaxdirs = compute_ridgedir(
            dirs,
            dir_scheme=dir_scheme,
            valids=valids,
        )
    orders = compute_flow_strahler_order(
        bmaxdirs,
        dir_scheme=dir_scheme,
        indegs=indegs,
        backend=backend,
    )
    return orders.astype(np.uint8, order="F")
