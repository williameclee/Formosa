# Last modified
#   2026-02-11, En-Chi Lee (williameclee@arizona.edu)
#     - Rename flowdir functions to be more descriptive
#   2026-06-09, En-Chi Lee (williameclee@gmail.com)
#     - Added `compute_flow_dist2ridge` function to compute 'distance to ridges'
#     - Added error for missing Fortran backend
#     - Removed Numpy type `np.bool` to either `np.bool_` or `bool` for compatibility with newer Numpy versions
#     - Renamed Fortran function call: `compute_masked_flowdir` -> `compute_synthetic_flowdir`
#     - Added `valids` argument to `label_flats` function
#   2026-06-10, En-Chi Lee (williameclee@gmail.com)
#     - Small refactors and documentation cleanup
#   2026-06-11, En-Chi Lee (williameclee@gmail.com)
#     - Moved Python backend implementations to separate file

import numpy as np

from formosa.geomorphology.d8directions import D8Directions

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


def _compute_flowdir_simple_py(
    dem: npt.NDArray[np.number],
    directions: D8Directions = D8Directions(),
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.bool_]]:
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
    valids: Optional[npt.NDArray[np.bool_]] = None,
    backend: Literal["fortran", "python"] = "fortran",
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.bool_]]:
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
            if valids is None:
                valids = np.ones(dem.shape, dtype=bool, order="F")
            flowdir, is_flat = flowdir_f.compute_flowdir_simple(
                dem.astype(np.float32, order="F"),
                valids.astype(bool, order="F"),
                directions.offsets.astype(np.int32, order="F"),
                directions.codes.astype(np.uint8, order="F"),
            )
    return flowdir.astype(np.uint8, order="F"), is_flat.astype(bool, order="F")


def _find_flat_edges_py(
    dem: npt.NDArray[np.number],
    flowdir: npt.NDArray[np.integer],
    directions=D8Directions(),
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    neighbours, _, _ = get_neighbour_values(
        dem,
        directions=directions,
        include_self=False,
        pad_value=np.min(dem) - 1,  # since is_high_edge
    )
    neighbour_flowdirs, _, _ = get_neighbour_values(
        flowdir, directions=directions, include_self=False, pad_value=-1
    )

    is_high_edge: npt.NDArray[np.bool_] = (flowdir == 0) & np.any(
        dem < neighbours, axis=0
    )
    is_low_edge: npt.NDArray[np.bool_] = (flowdir != 0) & (
        np.any((neighbour_flowdirs == 0) & (dem == neighbours), axis=0)
    )

    return is_low_edge, is_high_edge


def find_flat_edges(
    dem: npt.NDArray[np.number],
    flowdir: npt.NDArray[np.integer],
    directions: D8Directions = D8Directions(),
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
            if valids is None:
                valids = np.ones(dem.shape, dtype=bool, order="F")

            is_low_edge, is_high_edge = flowdir_f.find_flat_edges(
                dem.astype(np.float32, order="F"),
                flowdir.astype(np.int32, order="F"),
                valids.astype(bool, order="F"),
                directions.offsets.astype(np.int32, order="F"),
                directions.codes.astype(np.uint8, order="F"),
            )

    return (
        is_low_edge.astype(bool, order="F"),
        is_high_edge.astype(bool, order="F"),
    )


def label_flats(
    dem: npt.NDArray[np.number],
    seeds: npt.NDArray[np.bool_],
    valids: Optional[npt.NDArray[np.bool_]] = None,
    directions: D8Directions = D8Directions(),
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


def find_ambiguous(
    dem: npt.NDArray[np.number],
    directions: D8Directions = D8Directions(),
) -> npt.NDArray[np.bool_]:
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
    valids: Optional[npt.NDArray[np.bool_]] = None,
    only_min: bool = True,
    directions: D8Directions = D8Directions(window=3),
) -> npt.NDArray[np.bool_]:
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
    if valids is not None and np.any(~valids):
        dem[~valids] = np.max(dem[~valids]) + 1

    neighbours, _, _ = get_neighbour_values(
        dem, directions=directions, pad_value=np.nan, include_self=False
    )
    if only_min:
        is_flat = dem == np.nanmin(neighbours, axis=0)
    else:
        is_flat = np.any(dem == neighbours, axis=0)

    if valids is not None:
        is_flat = is_flat & valids
    return is_flat


def compute_away_from_high(
    labels: npt.NDArray[np.number],
    high_edges: npt.NDArray[np.bool_],
    directions: D8Directions = D8Directions(),
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
    directions : D8Directions, optional
        An instance of D8Directions defining the flow direction scheme, here it is used to determine the offsets for neighbor cells.
        Default is D8Directions().

    Returns
    -------
    NDArray[int32]
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

    z_syn = flowdir_f.away_from_high(
        labels.astype(np.int32, order="F"),
        high_edges.astype(bool, order="F"),
        directions.offsets.astype(np.int32, order="F"),
    )
    return z_syn.astype(np.int32, order="F")


def compute_towards_low(
    labels: npt.NDArray[np.number],
    low_edges: npt.NDArray[np.bool_],
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
    z_syn = flowdir_f.towards_low(
        labels.astype(np.int32, order="F"),
        low_edges.astype(bool, order="F"),
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
            flowdir = flowdir_f.compute_synthetic_flowdir(
                z.astype(np.int32, order="F"),
                labels.astype(np.int32, order="F"),
                directions.offsets.astype(np.int32, order="F"),
                directions.codes.astype(np.uint8, order="F"),
            )

    return flowdir.astype(np.uint8, order="F")


def _compute_flowdir_total(
    dem: npt.NDArray[np.number],
    directions: D8Directions = D8Directions(),
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
    directions : D8Directions, optional
        An instance of D8Directions defining the flow direction scheme.
        Default is D8Directions().
    valids : NDArray[bool], optional
        A boolean mask array indicating valid cells in the DEM.
        If None, all cells are considered valid.
        Default is None.
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
    flowdir : NDArray[uint8]
        A 2D integer array representing the flow directions for each cell in the DEM.
    is_flat : NDArray[bool]
        A boolean mask array where True indicates cells that are part of flat areas.
    flat_gradient : NDArray[int] | None
        A 2D integer array representing the synthetic elevation that resolves flat areas, or None if resolve_flat is False.
    """
    if fill_depression:
        dem = fill_depressions(dem, valids=valids, method=fill_depression_method)
    if resolve_flat:
        flowdir, is_flat, flat_gradient = _compute_flowdir_total(
            dem, directions=directions, valids=valids, step_size=step_size
        )
    else:
        flowdir, is_flat = compute_flowdir_simple(
            dem, directions=directions, valids=valids
        )
        flat_gradient = None
    return (
        flowdir.astype(np.uint8, order="F"),
        is_flat.astype(bool, order="F"),
        flat_gradient,
    )


def compute_indegree(
    flowdirs: npt.NDArray[np.integer],
    directions: D8Directions = D8Directions(),
    backend: Literal["fortran", "python"] = "fortran",
) -> npt.NDArray[np.int8]:
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
        Note: the Python implementation is unmaintained.
        Default is 'fortran'.

    Returns
    -------
    indegree : NDArray[int]
        A 2D integer array representing the indegree (number of upstream cells) for each cell.
    """
    match backend:
        case "python":
            from .flowdir_py import _compute_indegree_py

            indegree = _compute_indegree_py(flowdirs, directions=directions)
        case "fortran":
            indegree = flowdir_f.compute_indegree(
                flowdirs.astype(np.uint8, order="F"),
                directions.offsets.astype(np.int32, order="F"),
                directions.codes.astype(np.uint8, order="F"),
            )

    return indegree.astype(np.int8, order="F")


def compute_downstream_indices(
    *args, **kwargs
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
    from .flowdir_py import _compute_downstream_indices_py

    return _compute_downstream_indices_py(*args, **kwargs)


def compute_flowdir_graph(
    flowdirs: npt.NDArray[np.integer],
    valids: Optional[npt.NDArray[np.bool_]] = None,
    directions: D8Directions = D8Directions(),
    x: Optional[npt.NDArray[np.number]] = None,
    y: Optional[npt.NDArray[np.number]] = None,
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
    if valids is not None:
        assert (
            valids.shape == flowdirs.shape
        ), f"Shape for FLOWDIR and VALID mask must match, but got valid shape {flowdirs.shape} and flowdir shape {valids.shape} instead"
    else:
        valids = np.full(flowdirs.shape, True, dtype=bool)

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
            i[valids],
            dsi[valids],
            np.full(i[valids].size, np.nan),
        ),
        axis=1,
    ).ravel(order="C")
    graphj = np.stack(
        (
            j[valids],
            dsj[valids],
            np.full(j[valids].size, np.nan),
        ),
        axis=1,
    ).ravel(order="C")
    return graphi, graphj


def compute_flow_accumulation(
    flowdirs: npt.NDArray[np.integer],
    valids: Optional[npt.NDArray[np.bool_]] = None,
    weights: Optional[npt.NDArray[np.floating]] = None,
    indegrees: Optional[npt.NDArray[np.integer]] = None,
    dsij: Optional[npt.NDArray[np.integer]] = None,
    directions: D8Directions = D8Directions(),
    backend: Literal["fortran", "python"] = "fortran",
) -> npt.NDArray[np.float32]:
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
    accumulation : NDArray[float32]
        A 2D array representing the flow accumulation for each cell.
    """
    match backend:
        case "python":
            from .flowdir_py import _compute_flow_accumulation_py

            accumulation = _compute_flow_accumulation_py(
                flowdirs,
                valids=valids,
                weights=weights,
                indegrees=indegrees,
                dsij=dsij,
                directions=directions,
            )
        case "fortran":
            if indegrees is None:
                indegrees = compute_indegree(flowdirs, directions=directions)

            if valids is None:
                valids = np.ones(flowdirs.shape, dtype=bool)

            if weights is None:
                weights = np.where(valids, 1.0, 0.0).astype(np.float32)

            accumulation = flowdir_f.compute_flow_accumulation(
                flowdirs.astype(np.uint8, order="F"),
                valids.astype(bool, order="F"),
                weights.astype(np.float32, order="F"),
                indegrees.astype(np.int8, order="F"),
                directions.offsets.astype(np.int32, order="F"),
                directions.codes.astype(np.uint8, order="F"),
            )

    return accumulation.astype(np.float32, order="F")


def compute_strahler_order(
    flowdir: npt.NDArray[np.integer],
    directions: D8Directions = D8Directions(),
    valids: Optional[npt.NDArray[np.bool_]] = None,
    indegrees: Optional[npt.NDArray[np.integer]] = None,
    backend: Literal["fortran", "python"] = "fortran",
) -> npt.NDArray[np.int16]:
    """
    Computes the Strahler order for each cell in a flow direction grid.

    Parameters
    ----------
    flowdir : NDArray[int], optional
        A 2D array representing the flow directions for each cell.
        If None, indegrees and downstream_ij must be provided.
        Default is None.
    directions : D8Directions, optional
        An instance of D8Directions defining the flow direction scheme.
        Default is D8Directions().
    indegrees : NDArray[int], optional
        A 2D array representing the indegree (number of upstream cells) for each cell.
        If None, it will be computed from the flow direction grid.
        Default is None.
    backend : {'fortran', 'python'}, optional
        The backend to use for computation. 'fortran' uses the Fortran extension for performance, while 'python' uses a pure Python implementation.
        Default is 'fortran'.

    Returns
    -------
    strahler_order : NDArray[int16]
        A 2D integer array representing the Strahler order for each cell.
    """
    match backend:
        case "python":
            from .flowdir_py import _compute_strahler_order_py

            strahler_order = _compute_strahler_order_py(
                flowdir=flowdir,
                directions=directions,
                indegrees=indegrees,
            )
        case "fortran":
            if valids is None:
                valids = np.ones(flowdir.shape, dtype=bool)

            if indegrees is None:
                indegrees = compute_indegree(
                    flowdir, directions=directions, backend="fortran"
                )

            strahler_order = flowdir_f.compute_strahler_order(
                flowdir.astype(np.uint8, order="F"),
                valids.astype(bool, order="F"),
                indegrees.astype(np.int8, order="F"),
                directions.offsets.astype(np.int32, order="F"),
                directions.codes.astype(np.uint8, order="F"),
            )
            strahler_order[~valids] = 0
    return strahler_order.astype(np.int16, order="F")


def compute_flow_dist2source(
    flowdir: npt.NDArray[np.integer],
    directions: D8Directions = D8Directions(),
    x: Optional[npt.NDArray[np.number]] = None,
    y: Optional[npt.NDArray[np.number]] = None,
    valids: Optional[npt.NDArray[np.bool_]] = None,
    indegrees: Optional[npt.NDArray[np.integer]] = None,
) -> npt.NDArray[np.float32]:
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
    distance : NDArray[float32]
        A 2D array representing the downstream distance for each cell.

    Raises
    ------
    TypeError
        If the input arrays are not of the expected type or format.
    ValueError
        If the shapes of the input arrays do not match the expected dimensions.
    """
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

    distance = flowdir_f.compute_dist2source(
        flowdir.astype(np.uint8, order="F"),
        valids.astype(bool, order="F"),
        x.astype(np.float32, order="F"),
        y.astype(np.float32, order="F"),
        indegrees.astype(np.int8, order="F"),
        directions.offsets.astype(np.int32, order="F"),
        directions.codes.astype(np.uint8, order="F"),
    )
    return distance.astype(np.float32, order="F")


def label_watersheds(
    flowdir: npt.NDArray[np.integer],
    directions: D8Directions = D8Directions(),
    valids: Optional[npt.NDArray[np.bool_]] = None,
    backend: Literal["fortran", "python"] = "fortran",
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
    watersheds : NDArray[int32]
        A 2D array where each watershed is labeled with a unique integer.
    """
    match backend:
        case "python":
            from .flowdir_py import _label_watersheds_py

            watersheds = _label_watersheds_py(
                flowdir=flowdir,
                directions=directions,
                valids=valids,
            )
        case "fortran":
            if valids is None:
                valids = np.ones(flowdir.shape, dtype=bool)
            elif isinstance(valids, np.ndarray):
                assert (
                    valids.shape == flowdir.shape
                ), f"Shape for flow direction ({flowdir.shape}) and valid mask ({valids.shape}) do not match."
                valids = valids.astype(bool, copy=False) & (~np.isnan(flowdir))
                flowdir = np.where(valids, flowdir, np.nan)
            else:
                raise TypeError(
                    f"Valid mask must be a NumPy array (got {type(valids)})."
                )

            watersheds = flowdir_f.label_watersheds(
                flowdir.astype(np.uint8, order="F"),
                valids.astype(bool, order="F"),
                directions.offsets.astype(np.int32, order="F"),
                directions.codes.astype(np.uint8, order="F"),
            )
    return watersheds.astype(np.int32, order="F")


def compute_flow_dist2sink(
    flowdir: npt.NDArray[np.integer],
    directions: D8Directions = D8Directions(),
    x: Optional[npt.NDArray[np.number]] = None,
    y: Optional[npt.NDArray[np.number]] = None,
    valids: Optional[npt.NDArray[np.bool_]] = None,
) -> npt.NDArray[np.float32]:
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
    if valids is None:
        valids = ~np.isnan(flowdir)
    elif isinstance(valids, np.ndarray):
        assert (
            valids.shape == flowdir.shape
        ), f"Shape for flow direction ({valids.shape}) and valid mask ({flowdir.shape}) do not match."
        valids = valids.astype(bool, copy=False) & (~np.isnan(flowdir))
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

    distance = flowdir_f.compute_flow_dist2sink(
        flowdir.astype(np.uint8, order="F"),
        x.astype(np.float32, order="F"),
        y.astype(np.float32, order="F"),
        valids.astype(bool, order="F"),
        directions.offsets.astype(np.int32, order="F"),
        directions.codes.astype(np.uint8, order="F"),
    )
    return distance.astype(np.float32, order="F")


def compute_flow_dist2conf_max(
    flowdirs: npt.NDArray[np.integer],
    valids: Optional[npt.NDArray[np.bool_]] = None,
    x: Optional[npt.NDArray[np.number]] = None,
    y: Optional[npt.NDArray[np.number]] = None,
    watershed_labels: Optional[npt.NDArray[np.integer]] = None,
    directions: D8Directions = D8Directions(),
) -> npt.NDArray[np.float32]:
    """
    Computes the maximum distance to confluence for each cell with its neighbours in the flow direction grid.
    If the cell does not share a confluence with any of its neighbours, the distance to sink is returned instead.
    This field can be used as an proxy for the ridge network, where cells with a larger distance to confluence are more likely to be part of the ridge network.
    See `compute_flow_dist2ridge` for computing the distance to ridge based on this field.

    Parameters
    ----------
    flowdirs : NDArray[uint8]
        A 2D array representing the flow directions for each cell.
    valids : NDArray[bool], optional
        A boolean mask array where True indicates valid cells. If None, all cells are considered valid.
        Default is None.
    x : NDArray[int | float], optional
        A 2D array representing the x-coordinates of each cell. If None, a default grid will be created.
        Default is None.
    y : NDArray[int | float], optional
        A 2D array representing the y-coordinates of each cell. If None, a default grid will be created.
        Default is None.
    watershed_labels : NDArray[int], optional
         A 2D array representing labels for different watersheds in the flow direction grid. Since celss in different watersheds do not share confluences, providing watershed labels can skip unnecessary comparisons.
        If None, all cells are assigned the same label.
        Default is None.
    directions : D8Directions, optional
        An instance of D8Directions defining the flow direction scheme.
        Default is D8Directions().

    Returns
    -------
    NDArray[float32]
        A 2D array representing the maximum distance to confluence for each cell.
    """
    if valids is None:
        valids = np.ones(flowdirs.shape, dtype=bool)
    elif isinstance(valids, np.ndarray):
        assert (
            valids.shape == flowdirs.shape
        ), f"Shape for flow direction ({flowdirs.shape}) and valid mask ({valids.shape}) do not match."
    else:
        raise TypeError(f"Valid mask must be a NumPy array (got {type(valids)}).")
    if x is not None and y is not None:
        assert (
            x.shape == flowdirs.shape and y.shape == flowdirs.shape
        ), f"Shapes for flow direction ({flowdirs.shape}) and x ({x.shape}) and y ({y.shape}) must match."
    else:
        x = np.arange(flowdirs.shape[1], dtype=np.float32)
        y = np.arange(flowdirs.shape[0], dtype=np.float32)
        x, y = np.meshgrid(x, y, indexing="xy")
    if watershed_labels is None:
        watershed_labels = np.ones(flowdirs.shape, dtype=np.int32)
    elif isinstance(watershed_labels, np.ndarray):
        assert (
            watershed_labels.shape == flowdirs.shape
        ), f"Shape for flow direction ({flowdirs.shape}) and labels ({watershed_labels.shape}) do not match."
    else:
        raise TypeError(f"Labels must be a NumPy array (got {type(watershed_labels)}).")

    bmax = flowdir_f.compute_max_branch_dist(
        flowdirs.astype(np.uint8, order="F"),
        valids.astype(bool, order="F"),
        x.astype(np.float32, order="F"),
        y.astype(np.float32, order="F"),
        watershed_labels.astype(np.int32, order="F"),
        directions.offsets.astype(np.int32, order="F"),
        directions.codes.astype(np.uint8, order="F"),
    )
    return bmax.astype(np.float32, order="F")


def compute_flow_dist2ridge(
    flowdirs: npt.NDArray[np.integer],
    valids: Optional[npt.NDArray[np.bool_]] = None,
    x: Optional[npt.NDArray[np.number]] = None,
    y: Optional[npt.NDArray[np.number]] = None,
    watershed_labels: Optional[npt.NDArray[np.integer]] = None,
    directions: D8Directions = D8Directions(),
) -> npt.NDArray[np.float32]:
    """
    Computes the 'distance to ridge' for each cell in the flow direction grid.
    The ridge network/intensity is defined as the maximum distance to confluence (see `compute_flow_dist2conf_max`), and the distance to ridge is computed as the downstream distance traversing the inverse of the intensity.

    Parameters
    ----------
    flowdirs : NDArray[int]
        A 2D array representing the flow directions for each cell.
    valids : NDArray[bool], optional
        A boolean mask array where True indicates valid cells. If None, all cells are considered valid.
        Default is None.
    x : NDArray[int | float], optional
        A 2D array representing the x-coordinates of each cell. If None, a default grid will be created.
        Default is None.
    y : NDArray[int | float], optional
        A 2D array representing the y-coordinates of each cell. If None, a default grid will be created.
        Default is None.
    watershed_labels : NDArray[int], optional
        A 2D array representing labels for different watersheds in the flow direction grid. Since celss in different watersheds do not share confluences, providing watershed labels can skip unnecessary comparisons.
        If None, all cells are assigned the same label.
        Default is None.
    directions : D8Directions, optional
        An instance of D8Directions defining the flow direction scheme.
        Default is D8Directions().

    Returns
    -------
    dist : NDArray[float32]
        A 2D array representing the distance to ridge for each cell.
    """
    bmax = compute_flow_dist2conf_max(
        flowdirs,
        valids=valids,
        x=x,
        y=y,
        watershed_labels=watershed_labels,
        directions=directions,
    )
    bmaxdir, _, _ = compute_flowdir(
        -bmax, directions=directions, valids=valids, fill_depression=True
    )
    bmaxdist = compute_flow_dist2source(bmaxdir, directions=directions, valids=valids)
    return bmaxdist
