import numpy as np

from formosa.geomorphology.d8directions import D8Directions

import numpy.typing as npt
from typing import Literal


def _compute_flowdir_simple_py(
    dem: npt.NDArray[np.number],
    directions: D8Directions = D8Directions(),
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.bool]]:
    neighbours, codes, _ = get_neighbour_values(
        dem, directions=directions, include_self=True, pad_value=np.max(dem) + 1
    )
    flow2self_code = np.where(np.all(directions.offsets == [0, 0], axis=1))[0][0]
    flowdir = np.full(dem.shape, flow2self_code, dtype=np.int32)
    # find where not all neighbours are nan
    valid_mask = ~np.all(np.isnan(neighbours), axis=0)
    flowdir[valid_mask] = np.nanargmin(neighbours[:, valid_mask], axis=0)

    flowdir = codes[flowdir].astype(np.int32)
    is_flat = flowdir == 0
    return flowdir, is_flat


def compute_flowdir_simple(
    dem: npt.NDArray[np.number],
    directions: D8Directions = D8Directions(),
    valids: npt.NDArray[np.bool] | None = None,
    backend: Literal["fortran", "python"] = "fortran",
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.bool]]:
    """
    Computes flow directions for a DEM using a simple D8 algorithm.

    Parameters
    ----------
    dem : NDArray[number]
        A 2D array representing the digital elevation model (DEM).
    directions : D8Directions, optional
        An instance of D8Directions defining the flow direction scheme.
        Default is D8Directions().
    valids : NDArray[bool], optional
        A boolean mask array indicating valid cells in the DEM.
        If None, all cells are considered valid.
        Default is None.
    backend : {'fortran', 'python'}, optional
        The backend to use for computation. 'fortran' uses the Fortran extension for performance, while 'python' uses a pure Python implementation.
        Default is 'fortran'.

    Returns
    -------
    flowdir : NDArray[int]
        A 2D integer array representing the flow directions for each cell in the DEM.
    is_flat : NDArray[bool]
        A boolean mask array where True indicates cells that are part of flat areas.
    """
    match backend:
        case "python":
            flowdir, is_flat = _compute_flowdir_simple_py(dem, directions=directions)
        case "fortran":
            from formosa.geomorphology.flowdir_f import compute_flowdir_simple_f

            if valids is None:
                valids = np.ones(dem.shape, dtype=bool, order="F")
            flowdir, is_flat = compute_flowdir_simple_f(
                dem.astype(np.float32, order="F"),
                valids.astype(np.bool, order="F"),
                directions.offsets.astype(np.int32, order="F"),
                directions.codes.astype(np.int32, order="F"),
            )
    return flowdir.astype(np.uint8, order="F"), is_flat.astype(np.bool, order="F")


def _find_flat_edges_py(
    dem: npt.NDArray[np.number],
    flowdir: npt.NDArray[np.integer],
    directions=D8Directions(),
) -> tuple[npt.NDArray[np.bool], npt.NDArray[np.bool]]:
    neighbours, _, _ = get_neighbour_values(
        dem,
        directions=directions,
        include_self=False,
        pad_value=np.min(dem) - 1,  # since is_high_edge
    )
    neighbour_flowdirs, _, _ = get_neighbour_values(
        flowdir, directions=directions, include_self=False, pad_value=-1
    )

    is_high_edge: npt.NDArray[np.bool] = (flowdir == 0) & np.any(
        dem < neighbours, axis=0
    )
    is_low_edge: npt.NDArray[np.bool] = (flowdir != 0) & (
        np.any((neighbour_flowdirs == 0) & (dem == neighbours), axis=0)
    )

    return is_low_edge, is_high_edge


def find_flat_edges(
    dem: npt.NDArray[np.number],
    flowdir: npt.NDArray[np.integer],
    directions=D8Directions(),
    valids: npt.NDArray[np.bool] | None = None,
    backend: Literal["fortran", "python"] = "fortran",
) -> tuple[npt.NDArray[np.bool], npt.NDArray[np.bool]]:
    """
    Finds the cells on the edges of flat areas that drain to lower terrain (low edges) and those that are adjacent to higher terrain (high edges).
    From [R. Barnes *et al.* (2014)](https://doi.org/10.1016/j.cageo.2013.01.009), Algorithm 3 (p. 133).

    Parameters
    ----------
    dem : NDArray[number]
        A 2D array representing the digital elevation model (DEM).
    flowdir : NDArray[integer]
        A 2D array representing the flow direction for each cell in the DEM.
    directions : D8Directions, optional
        An instance of D8Directions defining the flow direction scheme that `flowdir` uses.
        Default is D8Directions().
    valids : NDArray[bool], optional
        A boolean mask array indicating valid cells in the DEM.
        If None, all cells are considered valid.
        Default is None.
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
            is_low_edge, is_high_edge = _find_flat_edges_py(
                dem, flowdir, directions=directions
            )
        case "fortran":
            from formosa.geomorphology.flowdir_f import find_flat_edges_f

            if valids is None:
                valids = np.ones(dem.shape, dtype=bool, order="F")

            is_low_edge, is_high_edge = find_flat_edges_f(
                dem.astype(np.float32, order="F"),
                flowdir.astype(np.int32, order="F"),
                valids.astype(np.bool, order="F"),
                directions.offsets.astype(np.int32, order="F"),
                directions.codes.astype(np.int32, order="F"),
            )

    return is_low_edge.astype(np.bool, order="F"), is_high_edge.astype(
        np.bool, order="F"
    )


def label_flats(
    dem: npt.NDArray[np.number],
    seeds: npt.NDArray[np.bool],
    directions: D8Directions = D8Directions(),
) -> npt.NDArray[np.int32]:
    """
    Separates and labels inidividual flat areas in a DEM.
    From [R. Barnes *et al.* (2014)](https://doi.org/10.1016/j.cageo.2013.01.009), Algorithm 4 (p. 133).

    Parameters
    ----------
    dem : NDArray[number]
        A 2D array representing the digital elevation model (DEM).
    seeds : NDArray[bool] | NDArray[int] | Iterable[Iterable[int]]
        Either a boolean mask array indicating flat area locations, or a 2D integer array of coordinates, or an iterable of coordinate pairs.
    directions : D8Directions, optional
        An instance of D8Directions defining the flow direction scheme.
        Default is D8Directions().

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
    from formosa.geomorphology.flowdir_f import label_flats_f

    assert (
        dem.shape == seeds.shape
    ), f"Shapes for dem ({dem.shape}) and seeds ({seeds.shape}) do not match."

    labels = label_flats_f(
        dem.astype(np.float64, order="F"),
        seeds.astype(np.bool, order="F"),
        directions.offsets.astype(np.int32, order="F"),
    )

    return labels.astype(np.int32, order="F")


def get_neighbour_values(
    array: np.ndarray,
    directions: D8Directions = D8Directions(),
    pad_value: np.number | float | int = np.nan,
    include_self: bool = False,
    self_at_last: bool = False,
) -> tuple[np.ndarray, npt.NDArray[np.integer], npt.NDArray[np.integer]]:
    """
    Gets the values of neighbouring cells in an array based on specified directions.

    Parameters
    ----------
    array : NDArray
        A 2D array from which to extract neighbour values.
    directions : D8Directions, optional
        An instance of D8Directions defining the neighbour offsets.
        Default is D8Directions().
    pad_value : number | float | int, optional
        Value to use for padding the array edges (default is np.nan).
    include_self : bool, optional
        Whether to include the value of the cell itself as a neighbour (default is False).
    self_at_last : bool, optional
        If include_self is True, whether to place the self value at the end of the neighbour list (default is False).

    Returns
    -------
    neighbours : NDArray
        A 3D array where the first dimension corresponds to neighbour indices and the other two dimensions match the input array.
    codes : NDArray[int]
        A 1D array of direction codes corresponding to the neighbours.
    offsets : NDArray[int]
        A 2D array of offsets (di, dj) corresponding to the neighbours.
    """
    # Input validation and initialisation
    if np.issubdtype(array.dtype, np.integer) and pad_value is np.nan:
        Warning("Integer array does not support NaN padding, using max int instead")
        pad_value = np.iinfo(array.dtype).max

    # Main
    # get padding width from offset
    pad_width = np.max(abs(directions.offsets))
    array_padded = np.pad(
        array,
        pad_width=pad_width,
        mode="constant",
        constant_values=pad_value,
    )
    neighbours = np.zeros((len(directions.codes), *array.shape), dtype=array.dtype)
    offsets = np.zeros((len(directions.codes), 2), dtype=np.int16)
    for i_offset, [di, dj] in enumerate(directions.offsets.astype(np.int16)):
        offsets[i_offset, :] = [di, dj]
        neighbours[i_offset, :, :] = array_padded[
            pad_width + di : pad_width + di + array.shape[0],
            pad_width + dj : pad_width + dj + array.shape[1],
        ]

    codes = directions.codes
    if not include_self:
        # exclude self (first offset)
        self_id = np.where(np.all(directions.offsets == [0, 0], axis=1))[0][0]
        neighbours = np.delete(neighbours, self_id, axis=0)
        codes = np.delete(codes, self_id, axis=0)
        offsets = np.delete(offsets, self_id, axis=0)
    elif self_at_last:
        neighbours = np.roll(neighbours, -1, axis=0)
        codes = np.roll(codes, -1, axis=0)
        offsets = np.roll(offsets, -1, axis=0)
    return neighbours, codes, offsets


def compute_downstream_indices(
    flowdirs: npt.NDArray[np.integer],
    directions: D8Directions = D8Directions(),
    valids: npt.NDArray[np.bool] | None = None,
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer], npt.NDArray[np.int32]]:
    """
    Computes the downstream indices for each cell in a flow direction grid.
    
    Parameters
    ----------
    flowdirs : NDArray[int]
        A 2D array representing the flow directions for each cell.
    directions : D8Directions, optional
        An instance of D8Directions defining the flow direction scheme.
        Default is D8Directions().
    valids : NDArray[bool], optional
        A boolean mask array indicating valid cells in the flow direction grid.
        If None, all cells are considered valid.
        Default is None.
    
    Returns
    -------
    dsi : NDArray[int]
        A 2D array of downstream row indices for each cell.
    dsj : NDArray[int]
        A 2D array of downstream column indices for each cell.
    dsij : NDArray[int32]
        A 2D array of flattened downstream indices for each cell.
    """
    if valids is None:
        valids = ~np.isnan(flowdirs)
    elif isinstance(valids, np.ndarray):
        assert (
            valids.shape == flowdirs.shape
        ), f"Shapes for flow direction ({flowdirs.shape}) and valid mask ({valids.shape}) do not match."
    else:
        raise TypeError(
            f"Expected valids to be None or np.ndarray, got {type(valids)} instead."
        )

    I, J = flowdirs.shape
    ii, jj = np.meshgrid(
        np.arange(I, dtype=np.int32), np.arange(J, dtype=np.int32), indexing="ij"
    )
    di, dj = directions.code2d8offset(flowdirs)
    dsi = (ii.astype(np.int16) + (di).astype(np.int16)).astype(np.int16)
    dsj = (jj.astype(np.int16) + (dj).astype(np.int16)).astype(np.int16)
    dsij: npt.NDArray[np.int32] = dsj.astype(np.int32) * I + dsi.astype(np.int32)

    if np.any((dsi < 0) | (dsi >= I) | (dsj < 0) | (dsj >= J)):
        raise ValueError("Some downstream indices out of bounds")

    return dsi, dsj, dsij


def find_ambiguous(
    dem: npt.NDArray[np.number],
    directions: D8Directions = D8Directions(),
) -> npt.NDArray[np.bool]:
    """
    Detects ambiguous flow directions in a DEM, where multiple neighbouring cells have the same minimum elevation.
    
    Parameters
    ----------
    dem : NDArray[number]
        A 2D array representing the digital elevation model (DEM).
    directions : D8Directions, optional
        An instance of D8Directions defining the flow direction scheme.
        Default is D8Directions().
    
    Returns
    -------
    is_ambiguous : NDArray[bool]
        A boolean mask array where True indicates cells with ambiguous flow directions.
    """
    neighbours, _, _ = get_neighbour_values(dem, directions=directions)
    min_neighbours = np.min(neighbours, axis=0)
    is_ambiguous = np.sum(neighbours == min_neighbours, axis=0) > 1
    is_ambiguous = is_ambiguous & ~(find_flat(dem))
    return is_ambiguous


def find_flat(
    dem: npt.NDArray[np.number],
    valid: npt.NDArray[np.bool] | None = None,
    only_min: bool = True,
    directions: D8Directions = D8Directions(window=3),
) -> npt.NDArray[np.bool]:
    """
    Identifies flat areas in a DEM where cells have no lower neighbouring cells.
    
    Parameters
    ----------
    dem : NDArray[number]
        A 2D array representing the digital elevation model (DEM).
    valid : NDArray[bool], optional
        A boolean mask array indicating valid cells in the DEM.
        If None, all cells are considered valid.
        Default is None.
    only_min : bool, optional
        If True, only cells that are equal to the minimum of their neighbours are considered flat.
        If False, cells equal to any neighbour are considered flat.
        Default is True.
    directions : D8Directions, optional
        An instance of D8Directions defining the neighbour offsets.
        Default is D8Directions(window=3).
    
    Returns
    -------
    is_flat : NDArray[bool]
        A boolean mask array where True indicates cells that are part of flat areas.
    """
    if valid is not None and np.any(~valid):
        dem[~valid] = np.max(dem[~valid]) + 1

    neighbours, _, _ = get_neighbour_values(
        dem, directions=directions, pad_value=np.nan, include_self=False
    )
    if only_min:
        is_flat = dem == np.nanmin(neighbours, axis=0)
    else:
        is_flat = np.any(dem == neighbours, axis=0)

    if valid is not None:
        is_flat = is_flat & valid
    return is_flat


def compute_away_from_high(
    labels: npt.NDArray[np.number],
    high_edges: npt.NDArray[np.bool],
    directions: D8Directions = D8Directions(),
) -> npt.NDArray[np.integer]:
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
    directions : D8Directions, optional
        An instance of D8Directions defining the flow direction scheme, here it is used to determine the offsets for neighbor cells.
        Default is D8Directions().

    Returns
    -------
    NDArray[integer]
        A 2D integer array representing the synthetic elevation that increases away from high edges within each flat region.

    Raises
    ------
    TypeError
        If the input high_edges is not of the expected type or format.
    ValueError
        If the shapes of the input arrays do not match the expected dimensions.
    """
    from formosa.geomorphology.flowdir_f import away_from_high_loop_f

    z_syn = away_from_high_loop_f(
        labels.astype(np.int32, order="F"),
        high_edges.astype(np.bool, order="F"),
        directions.offsets.astype(np.int32, order="F"),
    )
    return z_syn


def compute_towards_low(
    labels: npt.NDArray[np.number],
    low_edges: npt.NDArray[np.bool],
    directions: D8Directions = D8Directions(),
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
    directions : D8Directions, optional
        An instance of D8Directions defining the flow direction scheme, here it is used to determine the offsets for neighbor cells.
        Default is D8Directions().

    Returns
    -------
    NDArray[integer]
        A 2D integer array representing the synthetic elevation that increases towards low edges within each flat region.

    Raises
    ------
    TypeError
        If the input low_edges is not of the expected type or format.
    ValueError
        If the shapes of the input arrays do not match the expected dimensions.
    """
    from formosa.geomorphology.flowdir_f import towards_low_loop_f

    z_syn = towards_low_loop_f(
        labels.astype(np.int32, order="F"),
        low_edges.astype(np.bool, order="F"),
        directions.offsets.astype(np.int32, order="F"),
    )
    return z_syn


def _compute_masked_flowdir_py(
    z: npt.NDArray[np.integer | np.floating],
    labels: npt.NDArray[np.integer],
    directions: D8Directions = D8Directions(),
) -> npt.NDArray[np.integer]:
    neighbours, codes, _ = get_neighbour_values(
        z,
        directions=directions,
        include_self=True,
        pad_value=z.max() + 1,
    )
    neighbour_labels, _, _ = get_neighbour_values(
        labels, directions=directions, include_self=True, pad_value=-1
    )
    # Mask neighbours that are not in the same flat
    neighbours = np.where(
        neighbour_labels != labels[np.newaxis, :, :], np.inf, neighbours
    )
    min_indices = np.argmin(neighbours, axis=0)
    flowdir = codes[min_indices]
    flowdir[labels == 0] = 0

    return flowdir


def compute_masked_flowdir(
    z: npt.NDArray[np.integer | np.floating],
    labels: npt.NDArray[np.integer],
    directions: D8Directions = D8Directions(),
    backend: Literal["fortran", "python"] = "fortran",
) -> npt.NDArray[np.integer]:
    """
    Computes flow directions within flat areas using synthetic elevation.
    Very similar to the naive flow direction computation, but only search within the same flat area.

    Parameters
    ----------
    z : NDArray[int | float]
        A 2D array representing the synthetic elevation within flat areas.
    labels : NDArray[int]
        A 2D array where each flat region is labeled with a unique integer.
    directions : D8Directions, optional
        An instance of D8Directions defining the flow direction scheme.
        Default is D8Directions().
    backend : {'fortran', 'python'}, optional
        The backend to use for computation. 'fortran' uses the Fortran extension for performance,
        while 'python' uses a pure Python implementation. Default is 'fortran'.

    Returns
    -------
    flowdir : NDArray[int]
        A 2D integer array representing the flow directions within flat areas.
    """
    match backend:
        case "python":
            flowdir = _compute_masked_flowdir_py(z, labels, directions=directions)
        case "fortran":
            from formosa.geomorphology.flowdir_f import compute_masked_flowdir_f

            flowdir = compute_masked_flowdir_f(
                z.astype(np.int32, order="F"),
                labels.astype(np.int32, order="F"),
                directions.offsets.astype(np.int32, order="F"),
                directions.codes.astype(np.int32, order="F"),
            )

    return flowdir


def _compute_flowdir_total(
    dem: npt.NDArray[np.number],
    directions: D8Directions = D8Directions(),
    valids: npt.NDArray[np.bool] | None = None,
    step_size: int = 4,
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.bool], npt.NDArray[np.integer]]:
    """
    Computes flow directions for a DEM, resolving flat areas using synthetic elevations.
    Combines simple flow direction computation with flat area resolution from [R. Barnes *et al.* (2014)](https://doi.org/10.1016/j.cageo.2013.01.009).

    Parameters
    ----------
    dem : NDArray[number]
        A 2D array representing the digital elevation model (DEM).
    directions : D8Directions, optional
        An instance of D8Directions defining the flow direction scheme.
        Default is D8Directions().
    valids : NDArray[bool], optional
        A boolean mask array indicating valid cells in the DEM.
        If None, all cells are considered valid.
        Default is None.
    step_size : int, optional
        The increment in synthetic elevation per step away from low edges to avoid ties when combined with the result of `compute_away_from_high`.
        Default is 4.

    Returns
    -------
    flowdir : NDArray[int]
        A 2D integer array representing the flow directions for each cell in the DEM.
    is_flat : NDArray[bool]
        A boolean mask array where True indicates cells that are part of flat areas.
    z_syn : NDArray[int]
        A 2D integer array representing the synthetic elevation that resolves flat areas.
    """
    if step_size <= 0:
        raise ValueError(f"Step size must be a positive integer (got {step_size}).")
    flowdir, is_flat = compute_flowdir_simple(dem, directions=directions, valids=valids)

    is_low_edge, is_high_edge = find_flat_edges(
        dem, flowdir, directions=directions, valids=valids
    )

    flat_labels = label_flats(dem, (is_low_edge | is_flat), directions=directions)

    is_high_edge = is_high_edge & (flat_labels != 0)

    z_syn_away = compute_away_from_high(
        flat_labels, is_high_edge, directions=directions
    )

    z_syn_towards = compute_towards_low(
        flat_labels,
        is_low_edge,
        directions=directions,
    )
    z_syn = z_syn_away + z_syn_towards * step_size

    flat_flowdir = compute_masked_flowdir(z_syn, flat_labels, directions=directions)

    flowdir[flowdir == 0] = flat_flowdir[flowdir == 0]
    return flowdir, is_flat, z_syn


def compute_flowdir(
    dem: npt.NDArray[np.number],
    directions: D8Directions = D8Directions(),
    valids: npt.NDArray[np.bool] | None = None,
    resolve_flat: bool = True,
    step_size: int = 4,
) -> tuple[
    npt.NDArray[np.integer], npt.NDArray[np.bool], npt.NDArray[np.integer] | None
]:
    """
    Computes flow directions for a DEM, optionally resolving flat areas.

    Parameters
    ----------
    dem : NDArray[number]
        A 2D array representing the digital elevation model (DEM).
    directions : D8Directions, optional
        An instance of D8Directions defining the flow direction scheme.
        Default is D8Directions().
    valids : NDArray[bool], optional
        A boolean mask array indicating valid cells in the DEM.
        If None, all cells are considered valid.
        Default is None.
    resolve_flat : bool, optional
        Whether to resolve flat areas using synthetic elevations.
        Default is True.
    step_size : int, optional
        The increment in synthetic elevation per step away from low edges to avoid ties when combining synthetic elevations.
        Default is 4.

    Returns
    -------
    flowdir : NDArray[int]
        A 2D integer array representing the flow directions for each cell in the DEM.
    is_flat : NDArray[bool]
        A boolean mask array where True indicates cells that are part of flat areas.
    flat_gradient : NDArray[int] | None
        A 2D integer array representing the synthetic elevation that resolves flat areas, or None if resolve_flat is False.
    """
    if resolve_flat:
        flowdir, is_flat, flat_gradient = _compute_flowdir_total(
            dem, directions=directions, valids=valids, step_size=step_size
        )
    else:
        flowdir, is_flat = compute_flowdir_simple(
            dem, directions=directions, valids=valids
        )
        flat_gradient = None
    return flowdir, is_flat, flat_gradient


def _compute_indegree_py(
    flowdirs: npt.NDArray[np.integer], directions: D8Directions = D8Directions()
) -> npt.NDArray[np.integer]:
    indegree = np.zeros(flowdirs.shape, dtype=np.int32)
    dsi, dsj, _ = compute_downstream_indices(flowdirs, directions=directions)

    for flowdir in np.unique(flowdirs):
        if flowdir == 0:
            continue
        is_Valid_ds = (
            (flowdirs == flowdir)
            & (dsi >= 0)
            & (dsi < flowdirs.shape[0])
            & (dsj >= 0)
            & (dsj < flowdirs.shape[1])
        )
        indegree[dsi[is_Valid_ds], dsj[is_Valid_ds]] += 1

    return indegree


def compute_indegree(
    flowdirs: npt.NDArray[np.integer],
    directions: D8Directions = D8Directions(),
    backend: Literal["fortran", "python"] = "fortran",
) -> npt.NDArray[np.uint8]:
    """
    Computes the number of upstream cells (indegree) for each cell in a flow direction grid.

    Parameters
    ----------
    flowdirs : NDArray[int]
        A 2D array representing the flow directions for each cell.
    directions : D8Directions, optional
        An instance of D8Directions defining the flow direction scheme.
        Default is D8Directions().
    backend : {'fortran', 'python'}, optional
        The backend to use for computation. 'fortran' uses the Fortran extension for performance, while 'python' uses a pure Python implementation.
        Default is 'fortran'.

    Returns
    -------
    indegree : NDArray[int]
        A 2D integer array representing the indegree (number of upstream cells) for each cell.
    """
    match backend:
        case "python":
            indegree = _compute_indegree_py(flowdirs, directions=directions)
        case "fortran":
            from formosa.geomorphology.flowdir_f import compute_indegree_f

            indegree = compute_indegree_f(
                flowdirs.astype(np.int32, order="F"),
                directions.offsets.astype(np.int32, order="F"),
                directions.codes.astype(np.int32, order="F"),
            )

    return indegree.astype(np.uint8, order="F")


def compute_flowdir_graph(
    flowdirs: npt.NDArray[np.integer],
    valid: npt.NDArray[np.bool] | None = None,
    directions: D8Directions = D8Directions(),
    x: npt.NDArray[np.number] | None = None,
    y: npt.NDArray[np.number] | None = None,
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]]:
    """
    Computes a graph representation of the flow directions in a flow direction grid.
    
    Parameters
    ----------
    flowdirs : NDArray[int]
        A 2D array representing the flow directions for each cell.
    valid : NDArray[bool], optional
        A boolean mask array indicating valid cells in the flow direction grid.
        If None, all cells are considered valid.
        Default is None.
    directions : D8Directions, optional
        An instance of D8Directions defining the flow direction scheme.
        Default is D8Directions().
    x : NDArray[number], optional
        A 2D array representing the x-coordinates of each cell.
        If provided, the graph will use these coordinates instead of grid indices.
        Default is None.
    y : NDArray[number], optional
        A 2D array representing the y-coordinates of each cell.
        If provided, the graph will use these coordinates instead of grid indices.
        Default is None.
    
    Returns
    -------
    graphi : NDArray[int]
        A 1D array representing the row indices of the graph edges.
    graphj : NDArray[int]
        A 1D array representing the column indices of the graph edges.
    """
    if valid is not None:
        assert (
            valid.shape == flowdirs.shape
        ), f"Shape for FLOWDIR and VALID mask must match, but got valid shape {flowdirs.shape} and flowdir shape {valid.shape} instead"
    else:
        valid = np.full(flowdirs.shape, True, dtype=np.bool)

    i, j = np.meshgrid(
        np.arange(flowdirs.shape[0], dtype=np.int32),
        np.arange(flowdirs.shape[1], dtype=np.int32),
        indexing="ij",
    )
    dsi, dsj, _ = compute_downstream_indices(flowdirs, directions=directions)

    if x is not None and y is not None:
        j, i = x, y

        # Map i,j to actual coordinates
        dsj, dsi = x[dsi, dsj], y[dsi, dsj]

    graphi = np.stack(
        (
            i[valid],
            dsi[valid],
            np.full(i[valid].size, np.nan),
        ),
        axis=1,
    ).ravel(order="C")
    graphj = np.stack(
        (
            j[valid],
            dsj[valid],
            np.full(j[valid].size, np.nan),
        ),
        axis=1,
    ).ravel(order="C")
    return graphi, graphj


def _compute_accumulation_py(
    flowdirs: npt.NDArray[np.integer],
    valids=None,
    weights=None,
    indegrees: npt.NDArray[np.integer] | None = None,
    dsij: npt.NDArray[np.integer] | None = None,
    directions: D8Directions = D8Directions(),
) -> np.ndarray:
    from collections import deque

    # Initialisation
    I, J = flowdirs.shape

    if indegrees is None:
        indegrees = _compute_indegree_py(flowdirs, directions=directions)
    else:
        assert (
            indegrees.shape == flowdirs.shape
        ), f"Shape for flowdir and indegree must match, but got indegree shape {indegrees.shape} and flowdir shape {flowdirs.shape} instead"

    if valids is None:
        valids = (flowdirs != 0) | (indegrees > 0)
    else:
        assert (
            valids.shape == flowdirs.shape
        ), f"Shape for flowidr and valid mask must match, but got valid shape {valids.shape} and flowdir shape {flowdirs.shape} instead"

    if weights is None:
        weights = np.where(valids, 1, 0).astype(np.uint64)
    else:
        assert (
            weights.shape == flowdirs.shape
        ), f"Shape for flowdir and weight must match, but got weight shape {weights.shape} and flowdir shape {flowdirs.shape} instead"
        weights = np.where(valids, weights, 0)

    if dsij is None:
        _, _, dsij = compute_downstream_indices(flowdirs, directions=directions)
    else:
        assert (
            dsij.shape == flowdirs.shape
        ), f"Shape for flowdir and downstream ij indices must match, but got dsij: {dsij.shape} and flowdir: {flowdirs.shape} instead"

    indegrees = indegrees.flatten(order="F")
    valids = valids.flatten(order="F")
    weights = weights.flatten(order="F")
    dsij = dsij.flatten(order="F")
    flowdirs = flowdirs.flatten(order="F")

    # Initialize accumulation with self weight
    accumulation = weights.ravel().astype(weights.dtype, copy=True)

    # Queue sources (indeg == 0) among valid cells
    q = deque(np.flatnonzero((indegrees == 0) & valids))

    # Topological propagation
    while q:
        u = q.popleft()
        v = dsij[u]
        if not valids[v]:
            continue
        accumulation[v] += accumulation[u]
        indegrees[v] -= 1
        if indegrees[v] == 0:
            q.append(v)

    accumulation = accumulation.reshape(I, J, order="F")

    return accumulation


def compute_accumulation(
    flowdirs: npt.NDArray[np.integer],
    valids: npt.NDArray[np.bool] | None = None,
    weights: npt.NDArray[np.floating] | None = None,
    indegrees: npt.NDArray[np.integer] | None = None,
    dsij: npt.NDArray[np.integer] | None = None,
    directions: D8Directions = D8Directions(),
    backend: Literal["fortran", "python"] = "fortran",
) -> npt.NDArray[np.float64]:
    """
    Computes flow accumulation for each cell in a flow direction grid.

    Parameters
    ----------
    flowdirs : NDArray[int]
        A 2D array representing the flow directions for each cell.
    valids : NDArray[bool], optional
        A boolean mask array indicating valid cells in the flow direction grid.
        If None, all cells are considered valid.
        Default is None.
    weights : NDArray[float], optional
        A 2D array of weights for each cell, representing the contribution of each cell to its downstream cell.
        If None, each valid cell contributes a weight of 1.0.
        Default is None.
    indegrees : NDArray[int], optional
        A 2D array representing the indegree (number of upstream cells) for each cell.
        If None, indegrees are computed from the flow direction grid.
        Default is None.
    dsij : NDArray[int], optional
        A 2D array of downstream cell indices for each cell.
        If None, downstream indices are computed from the flow direction grid.
        Default is None.
    directions : D8Directions, optional
        An instance of D8Directions defining the flow direction scheme.
        Default is D8Directions().
    backend : {'fortran', 'python'}, optional
        The backend to use for computation. 'fortran' uses the Fortran extension for performance, while 'python' uses a pure Python implementation.
        Default is 'fortran'.

    Returns
    -------
    accumulation : NDArray[float64]
        A 2D array representing the flow accumulation for each cell.
    """
    match backend:
        case "python":
            accumulation = _compute_accumulation_py(
                flowdirs,
                valids=valids,
                weights=weights,
                indegrees=indegrees,
                dsij=dsij,
                directions=directions,
            )
        case "fortran":
            from formosa.geomorphology.flowdir_f import compute_accumulation_f

            if indegrees is None:
                indegrees = compute_indegree(flowdirs, directions=directions)

            if valids is None:
                valids = np.ones(flowdirs.shape, dtype=bool)

            if weights is None:
                weights = np.where(valids, 1.0, 0.0).astype(np.float64)

            accumulation = compute_accumulation_f(
                flowdirs.astype(np.int32, order="F"),
                valids.astype(np.bool, order="F"),
                weights.astype(np.float64, order="F"),
                indegrees.astype(np.int32, order="F"),
                directions.offsets.astype(np.int32, order="F"),
                directions.codes.astype(np.int32, order="F"),
            )

    return accumulation.astype(np.float64, order="F")


def compute_strahler_order(
    flowdir: npt.NDArray[np.integer] | None = None,
    directions: D8Directions = D8Directions(),
    indegrees: npt.NDArray[np.integer] | None = None,
    downstream_ij: npt.NDArray[np.integer] | None = None,
) -> npt.NDArray[np.integer]:
    from collections import deque

    if flowdir is None and (indegrees is None or downstream_ij is None):
        raise ValueError(
            "[FORMOSA] Either FLOWDIR or (INDEGREES and DOWNSTREAM_IJ) must be provided"
        )
    elif (indegrees is None or downstream_ij is None) and flowdir is not None:
        downstream_i, downstreamj, _ = compute_downstream_indices(
            flowdir, directions=directions
        )
        indegrees = _compute_indegree_py(flowdir, directions=directions)
    else:
        raise NotImplementedError("Unknown case for FLOWDIR and INDEGREES")

    strahler_order = np.zeros(indegrees.shape, dtype=np.int32)
    strahler_order[indegrees == 0] = 1

    ii, jj = np.indices(indegrees.shape, dtype=np.int32)
    seeds = deque(zip(ii[indegrees == 0], jj[indegrees == 0]))  # type: ignore TODO: figure out what the type error actually is

    while seeds:
        ci, cj = seeds.popleft()
        dsi, dsj = (
            downstream_i[ci, cj],
            downstreamj[ci, cj],
        )
        if (ci, cj) == (dsi, dsj):
            continue
        if strahler_order[dsi, dsj] < strahler_order[ci, cj]:
            strahler_order[dsi, dsj] = strahler_order[ci, cj]
        else:
            strahler_order[dsi, dsj] += 1
        indegrees[dsi, dsj] -= 1
        if indegrees[dsi, dsj] == 0:
            seeds.append((dsi, dsj))
    return strahler_order


def compute_flow_distance(
    flowdir: npt.NDArray[np.integer],
    directions: D8Directions = D8Directions(),
    x: npt.NDArray[np.integer | np.floating] | None = None,
    y: npt.NDArray[np.integer | np.floating] | None = None,
    valids: npt.NDArray[np.bool] | None = None,
    indegrees: npt.NDArray[np.integer] | None = None,
) -> npt.NDArray[np.float64]:
    """
    Computes the distance downstream along flow directions for each cell in the flow direction grid.

    Parameters
    ----------
    flowdir : NDArray[int]
        A 2D array representing the flow direction for each cell.
    directions : D8Directions, optional
        An instance of D8Directions defining the flow direction scheme.
        Default is D8Directions().
    x : NDArray[int | float], optional
        A 2D array representing the x-coordinates of each cell. If None, cell indices are used.
        Default is None.
    y : NDArray[int | float], optional
        A 2D array representing the y-coordinates of each cell. If None, cell indices are used.
        Default is None.
    valids : NDArray[bool], optional
        A boolean mask array indicating valid cells in the flow direction grid.
        If None, all cells are considered valid.
        Default is None.
    indegrees : NDArray[int], optional
        A 2D array representing the indegree (number of upstream cells) for each cell.
        If None, indegrees are computed from the flow direction grid.
        Default is None.

    Returns
    -------
    distance : NDArray[float64]
        A 2D array representing the downstream distance for each cell.

    Raises
    ------
    TypeError
        If the input arrays are not of the expected type or format.
    ValueError
        If the shapes of the input arrays do not match the expected dimensions.
    """
    from formosa.geomorphology.flowdir_f import compute_distance_f

    if valids is None:
        valids = np.ones(flowdir.shape, dtype=bool)
    elif isinstance(valids, np.ndarray):
        assert (
            valids.shape == flowdir.shape
        ), f"Shape for flow direction ({flowdir.shape}) and valid mask ({valids.shape}) do not match."
    else:
        raise TypeError(f"Valid mask must be a NumPy array (got {type(valids)}).")
    if x is not None and y is not None:
        assert (
            x.shape == flowdir.shape and y.shape == flowdir.shape
        ), f"Shapes for flow direction ({flowdir.shape}) and x ({x.shape}) and y ({y.shape}) must match."
    else:
        x = np.arange(flowdir.shape[1], dtype=np.float32)
        y = np.arange(flowdir.shape[0], dtype=np.float32)
        x, y = np.meshgrid(x, y, indexing="xy")
    if indegrees is None:
        indegrees = compute_indegree(flowdir, directions=directions)
    elif isinstance(indegrees, np.ndarray):
        assert (
            indegrees.shape == flowdir.shape
        ), f"Shape for flow direction ({flowdir.shape}) and indegree ({indegrees.shape}) do not match."
    else:
        raise TypeError(f"Indegree must be a NumPy array (got {type(indegrees)}).")

    distance = compute_distance_f(
        flowdir.astype(np.int32, order="F"),
        valids.astype(np.bool, order="F"),
        x.astype(np.float32, order="F"),
        y.astype(np.float32, order="F"),
        indegrees.astype(np.int32, order="F"),
        directions.offsets.astype(np.int32, order="F"),
        directions.codes.astype(np.int32, order="F"),
    )
    return distance.astype(np.float64, order="F")


def label_watersheds(
    flowdir: npt.NDArray[np.integer],
    directions: D8Directions = D8Directions(),
    valids: npt.NDArray[np.bool] | None = None,
) -> npt.NDArray[np.int32]:
    """
    Finds and labels watersheds in a DEM based on flow direction.
    
    Parameters
    ----------
    flowdir : NDArray[int]
        A 2D array representing the flow direction for each cell.
    directions : D8Directions, optional
        An instance of D8Directions defining the flow direction scheme.
        Default is D8Directions().
    valids : NDArray[bool], optional
        A boolean mask array indicating valid cells in the flow direction grid.
        If None, all non-NaN cells in flowdir are considered valid.
        Default is None.
    
    Returns
    -------
    watershed : NDArray[int32]
        A 2D array where each watershed is labeled with a unique integer.
    """
    if valids is None:
        valids = ~np.isnan(flowdir)
    elif isinstance(valids, np.ndarray):
        assert (
            valids.shape == flowdir.shape
        ), f"Shape for flow direction ({valids.shape}) and valid mask ({flowdir.shape}) do not match."
        valids = valids.astype(np.bool, copy=False) & (~np.isnan(flowdir))
        flowdir = np.where(valids, flowdir, np.nan)
    else:
        raise TypeError(
            f"[FORMOSA] VALIDS must be either None or a numpy array, got {type(valids)} instead."
        )

    I, J = flowdir.shape
    ii, jj = np.meshgrid(
        np.arange(I, dtype=np.int32), np.arange(J, dtype=np.int32), indexing="ij"
    )
    codes: list[int] = directions.codes.tolist()
    offsets: list[tuple[int, int]] = [
        (int(di), int(dj)) for di, dj in directions.offsets.astype(np.int32, copy=False)
    ]

    seeds: list[tuple[int, int]] = list(
        zip(ii[valids & (flowdir == 0)], jj[valids & (flowdir == 0)])
    )

    watershed = -np.ones(flowdir.shape, dtype=np.int32)

    for label, seed in enumerate(seeds):
        to_fill: list[tuple[int, int]] = [seed]

        while to_fill:
            ci, cj = to_fill.pop(0)
            watershed[ci, cj] = label
            for code, (di, dj) in zip(codes, offsets):
                ni, nj = ci - di, cj - dj
                if (ni < 0 or ni >= I) or (nj < 0 or nj >= J):
                    continue
                elif not valids[ni, nj]:
                    continue
                elif watershed[ni, nj] != -1:
                    continue

                if flowdir[ni, nj] == code:
                    to_fill.append((ni, nj))
    watershed = watershed + 1  # make background 0 and watersheds start from 1
    return watershed


def compute_back_distance(
    flowdir: npt.NDArray[np.integer],
    directions: D8Directions = D8Directions(),
    x: npt.NDArray[np.integer | np.floating] | None = None,
    y: npt.NDArray[np.integer | np.floating] | None = None,
    valids: npt.NDArray[np.bool] | None = None,
) -> npt.NDArray[np.float64]:
    """
    Computes the distance upstream along flow directions for each cell in the flow direction grid.

    Parameters
    ----------
    flowdir : NDArray[int]
        A 2D array representing the flow direction for each cell.
    directions : D8Directions, optional
        An instance of D8Directions defining the flow direction scheme.
        Default is D8Directions().
    x : NDArray[int | float], optional
        A 2D array representing the x-coordinates of each cell. If None, a default grid will be created.
    y : NDArray[int | float], optional
        A 2D array representing the y-coordinates of each cell. If None, a default grid will be created.
    valids : NDArray[bool], optional
        A boolean mask array where True indicates valid cells. If None, all non-NaN cells in flowdir are considered valid.

    Returns
    -------
    NDArray[float32]
        A 2D array representing the upstream distance for each cell.
    """
    from formosa.geomorphology.flowdir_f import compute_back_distance_f

    if valids is None:
        valids = ~np.isnan(flowdir)
    elif isinstance(valids, np.ndarray):
        assert (
            valids.shape == flowdir.shape
        ), f"Shape for flow direction ({valids.shape}) and valid mask ({flowdir.shape}) do not match."
        valids = valids.astype(np.bool, copy=False) & (~np.isnan(flowdir))
        flowdir = np.where(valids, flowdir, np.nan)
    else:
        raise TypeError(
            f"Validity mask must be either None or a numpy array, (got {type(valids)})."
        )
    if x is not None and y is not None:
        assert (
            x.shape == flowdir.shape and y.shape == flowdir.shape
        ), f"Shapes for flow direction ({flowdir.shape}) and x ({x.shape}) and y ({y.shape}) must match."
    else:
        x = np.arange(flowdir.shape[1], dtype=np.float32)
        y = np.arange(flowdir.shape[0], dtype=np.float32)
        x, y = np.meshgrid(x, y, indexing="xy")

    distance: npt.NDArray[np.float32] = compute_back_distance_f(
        flowdir.astype(np.int32, order="F"),
        x.astype(np.float32, order="F"),
        y.astype(np.float32, order="F"),
        valids.astype(np.bool, order="F"),
        directions.offsets.astype(np.int32, order="F"),
        directions.codes.astype(np.int32, order="F"),
    )
    return distance.astype(np.float64, order="F")
