# Last modified
#   2026-07-01, En-Chi Lee (williameclee@gmail.com)
#     - Opted out of the out-of-bound check in `compute_downstream_indices` in `create_flowgraph`
#     - Added function `construct_flowgraph`
#   2026-07-09, En-Chi Lee (williameclee@gmail.com)
#     - Specified endpoint index definition for `construct_flowgraph`
#     - Implemented Fortran backend of function `simplify_flowgraph` and function `concat_flowgraph`
#     - Added vertex mask to output of function `simplify_flowgraph`
#   2026-07-12, En-Chi Lee (williameclee@gmail.com)
#     - Implemented Python and Fortran backends of function `locate_invalid_graph_topology`
#   2026-07-13, En-Chi Lee (williameclee@gmail.com)
#     - Added default topology check to `simplify_flowgraph`
#   2026-07-14, En-Chi Lee (williameclee@gmail.com)
#     - Added simultaneous multi-graph checks to `simplify_flowgraph`
#     - Updated variable names in `locate_invalid_graph_topology`
#     - Split `geomorphology.flowdir` into submodules
#   2026-07-27, En-Chi Lee (williameclee@gmail.com)
#     - Implemented `insert_endpt` and relevant helper functions
#   2026-07-28, En-Chi Lee (williameclee@gmail.com)
#     - Implemented `solve_graph_overlaps` and relevant helper functions
#     - Integrated overlap resolution into simultaneous multi-graph simplification
#   2026-07-29, En-Chi Lee (williameclee@gmail.com)
#     - Made topology intersection results complete using scan-and-retry
#     - Added validation to simplified graph before return
#   2026-07-30, En-Chi Lee (williameclee@gmail.com)
#     - Fixed Python/FORTRAN backend behaviour parity in `compute_flow_strahler_order`.


import numpy as np

from formosa.geomorphology.flowdir.d8directions import D8Directions
import formosa.geomorphology.flowdir.raster as raster
from formosa.geomorphology.flowdir.utils import compute_downstream_indices

try:
    from formosa.geomorphology.flowdir_f import flowdir_graphs as graphs_f
except ImportError as err:

    class _MissingFortranBackend:
        def __init__(self, err: ImportError):
            self._err = err

        def __getattr__(self, name):
            raise ImportError(
                "formosa.geomorphology.graphs_f is required for backend='fortran' but is not available."
            ) from self._err

    graphs_f = _MissingFortranBackend(err)

import warnings

import numpy.typing as npt
from typing import Literal, Iterable, Optional, overload


class GraphTopologyError(RuntimeError):
    """
    Base exception for a graph that fails topology validation.
    """

    def __init__(
        self,
        message: str,
        intersections: npt.NDArray[np.integer],
    ) -> None:
        super().__init__(message)
        self.intersections = np.asarray(intersections, dtype=np.int32).copy()


class InvalidOriginalGraphTopology(GraphTopologyError):
    """
    Raised when an invalid result originated from invalid input topology.
    """


class UnresolvedSimplificationTopology(GraphTopologyError):
    """
    Raised when simplification leaves invalid topology from valid input.
    """


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


def construct_flowgraph(
    dirs: npt.NDArray[np.integer],
    dir_scheme: D8Directions = D8Directions(),
    valids: Optional[npt.NDArray[np.bool_]] = None,
    min_order: int = 2,
    orders: Optional[npt.NDArray[np.integer]] = None,
    preserve_junctions: bool = True,
    sort: bool = True,
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
        Default is `True`.
    sort : bool, option
        Whether to sort the flow graph by arc order and then by length
        Default is `True`.
    backend : {'fortran', 'python'}, optional
        The backend to use for computation
        'fortran' uses the Fortran extension for performance, while 'python' uses a pure Python implementation.
        Default is 'fortran'.

    Returns
    -------
    arc_orders : NDArray[int8]
        1D array representing the Strahler order for each arc in the flow graph
    vertex_ijs : NDArray[int32]
        V-by-2 array containing the ordered (i, j) incices of all arcs, concactinated together
    vertex_endpts : NDArray[int32]
        A-by-2 array containing the indices of where each arc starts and ends in `vertex_ijs`
        The returned endpoints are inclusive, meaning slicing must be done as `vertex_ijs[start : end + 1]`.
    """
    if valids is None:
        valids = np.ones(dirs.shape, dtype=bool)
    if orders is None:
        orders = raster.compute_flow_strahler_order(
            dirs,
            dir_scheme=dir_scheme,
            valids=valids,
            backend=backend,
        )

    # Find seed cells to start with
    valids = valids & (orders >= min_order)
    ncells = int(np.sum(valids))
    indegs = raster.count_indegree(
        dirs, dir_scheme=dir_scheme, valids=valids, backend=backend
    )
    seeds = valids & (indegs == 0)

    match backend:
        case "python":
            from .graphs_py import _construct_flowgraph_py

            narcs, nvertices, arc_orders, vertex_ijs, arc_endpts = (
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
            narcs, nvertices, arc_orders, vertex_ijs, arc_endpts = (
                graphs_f.construct_flowgraph(
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
            arc_endpts -= 1

    arc_orders = arc_orders[:narcs].T.copy(order="C")
    arc_endpts = arc_endpts[:, :narcs].T.copy(order="C")
    vertex_ijs = vertex_ijs[:, :nvertices].T.copy(order="C")

    if sort:
        arc_lengths = arc_endpts[:, 1] - arc_endpts[:, 0] + 1
        id = np.lexsort((arc_lengths, arc_orders))
        arc_orders = arc_orders[id]
        arc_endpts = arc_endpts[id, :]

    return arc_orders, vertex_ijs, arc_endpts


def find_vertex_id(
    verts: npt.NDArray[np.number],
    vert: npt.NDArray[np.number],
    n: Optional[int] = None,
) -> int | list[int]:
    """
    Finds the index (or indices) of a vertex in a list of vertices.

    Parameters
    ----------
    verts : NDArray[int | float]
        V-by-m array representing the m-dimensional coordinates of the vertices.
    vert :  NDArray[int | float]
        m-by-(1) array representing the m-dimensional coordinate of the vertex to find.
    n : int, optional
        Maximum number of indices to return, if the vertex appears multiple times in the array.
        When not specified, all occurences are returned.
        Default value is `None`.

    Returns
    -------
    ivert : int | list[int]
        Index (or indices) of the vertex in the list of vertices

    Raises
    ------
    AssertionError
        If the dimension of the provided vertex does not match the dimension of the array of vertices
    ValueError
        If the provided vertex is not found in the list of vertices
    """

    assert np.size(vert, 0) == np.size(verts, 1), (
        "The vertex and vertex array must have the same number of dimensions, "
        + f"but got {np.size(vert, 0)} and {np.size(verts, 1)}, respectively, instead."
    )

    ivert = np.squeeze(np.where(np.all(verts == vert, axis=1)))
    if np.size(ivert) == 0:
        raise ValueError("Provided vertex is not found in the list of vertices.")
    elif np.size(ivert) > 1:
        if (n is not None) and (np.size(ivert) > n):
            return ivert[:n].tolist()
        return ivert.tolist()
    return int(ivert)


@overload
def find_arc_id_of_vertex(
    endpts: npt.NDArray[np.integer], ivert: int, is_inclusive: bool = True
) -> int:
    """
    Finds the index of the arc that contains the vertex of a given index.

    Parameters
    ----------
    endpts : NDArray[int]
        A-by-2 array containing the indices of the starting and ending endpoint of each arc in a vertex array.
    ivert : int
        Index of the vertex in a vertex array to find the arc for.
    is_inclusive : bool
        Whether the `endpts` array is inclusive or half-open.
        If it is inclusive, the corresponding vertices in the arc are start_id ... end_id; if half-open, the vertices are start_id ... end_id - 1 instead.
        Default is `True`.

    Returns
    -------
    iarc : int
        Index of the arc that contains the vertex of a given index.
    """
    ...


@overload
def find_arc_id_of_vertex(
    endpts: npt.NDArray[np.integer], ivert: Iterable[int], is_inclusive: bool = True
) -> list[int]:
    """
    Finds the indices of the arcs that contain the vertices of a list of given indices.

    Parameters
    ----------
    endpts : NDArray[int]
        A-by-2 array containing the indices of the starting and ending endpoint of each arc in a vertex array.
    ivert : Iterable[int]
        Indices of the vertices in a vertex array to find the arcs for.
    is_inclusive : bool
        Whether the `endpts` array is inclusive or half-open.
        If it is inclusive, the corresponding vertices in the arc are start_id ... end_id; if half-open, the vertices are start_id ... end_id - 1 instead.
        Default is `True`.

    Returns
    -------
    iarc : list[int]
        Indices of the arcs that contain the vertices of the given indices.
    """
    ...


def find_arc_id_of_vertex(
    endpts: npt.NDArray[np.integer],
    ivert: int | Iterable[int],
    is_inclusive: bool = True,
) -> Optional[int] | list[Optional[int]]:
    def _find_arc_of_vertex(
        endpts: npt.NDArray[np.integer], ivert: int, is_inclusive: bool = True
    ) -> int:
        iarc = np.flatnonzero(
            (ivert >= endpts[:, 0]) & (ivert <= (endpts[:, 1] - (not is_inclusive)))
        )
        if np.size(iarc) == 0:
            warnings.warn("Provided vertex is not a part of any arc.")
            return None
        elif np.size(iarc) > 1:
            raise ValueError("Provided vertex is found in multiple arcs.")
        return iarc[0]

    if isinstance(ivert, int) or (np.size(ivert) == 1):
        iarc = _find_arc_of_vertex(endpts, ivert, is_inclusive)
        return iarc
    iarc = [_find_arc_of_vertex(endpts, ivert, is_inclusive) for ivert in ivert]
    return iarc


def insert_endpt(
    orders: npt.NDArray[np.integer],
    ijs: npt.NDArray[np.number],
    endpts: npt.NDArray[np.integer],
    add_endpt: npt.NDArray[np.number] | int,
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.number], npt.NDArray[np.integer]]:
    """
    Turns an interior vertex of a flow graph in to an endpoint.

    Parameters
    ----------
    orders : NDArray[int]
        O-by-(1) array representing the Strahler order for each arc.
    ijs : NDArray[int | float]
        V-by-n array representing the coordinates of the vertices.
    endpts : NDArray[int]
        A-by-2 array representing the indices of the starting and ending endpoint of each arc in the `ijs` array.
        The endpoints should be inclusive.
    add_endpt : NDArray[int | float] | int
        Either:
         1. n-by-(1) array representing the coordinate of the vertex to turn to an endpoint
         2. Integer specifying the index of the vertex in the `ijs` array to turn to an endpoint

    Returns
    -------
    orders : NDArray[int]
        Strahler order for each arc in the updated flow graph.
    ijs : NDArray[int | float]
        Coordinates of the vertices in the updated flow graph.
    endpts : NDArray[int]
        Inclusive starting and ending vertex indices for each arc in the updated flow graph.

    Raises
    ------
    AssertionError
        If `orders` and `endpts` do not contain the same number of arcs, or if a
        coordinate supplied as `add_endpt` does not have the same dimensionality
        as the vertices in `ijs`.
    """

    assert np.size(orders, 0) == np.size(endpts, 0), (
        "The orders array must have the same length as the endpoints array, "
        + f"but got {np.size(orders, 0)} and {np.size(endpts, 0)}, respectively, instead."
    )

    if isinstance(add_endpt, int):
        ivert = add_endpt
    else:
        try:
            ivert = find_vertex_id(ijs, add_endpt)
        except (AssertionError, ValueError):
            warnings.warn(
                "Provided endpoint is not found in the list of vertices. "
                + "Returning the original graph."
            )
            return orders, ijs, endpts

        # Exclude matching coordinates stored outside the ranges used by any arc
        iverts = np.atleast_1d(ivert)
        useds = np.any(
            (iverts[:, np.newaxis] >= endpts[np.newaxis, :, 0])
            & (iverts[:, np.newaxis] <= endpts[np.newaxis, :, 1]),
            axis=1,
        )
        iverts = iverts[useds]
        if iverts.size == 0:
            warnings.warn(
                "Provided endpoint is not a part of any arc. "
                + "Returning the original graph."
            )
            return orders, ijs, endpts
        ivert = int(iverts[0]) if iverts.size == 1 else iverts.tolist()
    iarc = find_arc_id_of_vertex(endpts, ivert)

    def _insert_endpt(orders, ijs, endpts, iarc, ivert):
        # Skip if the additional endpoint is already an endpoint
        if (endpts[iarc, 0] == ivert) or (endpts[iarc, 1] == ivert):
            return orders, ijs, endpts

        # Append the second half of the segment
        start_vert = np.size(ijs, 0)
        ijs = np.concat([ijs, ijs[ivert : np.squeeze(endpts[iarc, 1] + 1), :]])
        end_vert = np.size(ijs, 0) - 1
        endpts = np.concat([endpts, np.array([[start_vert, end_vert]])])
        orders = np.concat([orders, orders[iarc : iarc + 1]])

        # Truncate the current segment to the first half
        endpts[iarc, 1] = ivert

        return orders, ijs, endpts

    if isinstance(ivert, int):
        orders, ijs, endpts = _insert_endpt(orders, ijs, endpts, iarc, ivert)
        return orders, ijs, endpts

    for jvert, jarc in zip(ivert, iarc):
        if jarc is None:
            continue
        orders, ijs, endpts = _insert_endpt(orders, ijs, endpts, jarc, jvert)
    return orders, ijs, endpts


def concat_flowgraph(
    arc_orders: npt.NDArray[np.integer],
    vertex_ijs: npt.NDArray[np.integer],
    arc_endpts: npt.NDArray[np.integer],
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer], npt.NDArray[np.integer]]:
    """
    Concactenate arcs of the same order in a flow graph, separated by NaNs.
    It mainly serves to reduce the number of drawing calls when visualising the graph.

    Parameters
    ----------
    arc_orders : NDArray[int]
        O-by-1 array representing the Strahler order for each arc in the flow graph
    vertex_ijs : NDArray[int]
        V-by-2 array containing the ordered (i, j) incices of all arcs, concactinated together
    vertex_startends : NDArray[int]
        A-by-2 array containing the indices of where each arc starts and ends in `vertex_ijs`
        The returned endpoints are inclusive, meaning slicing must be done as `vertex_ijs[start : end + 1]`.

    Returns
    ----------
    arc_orders : NDArray[int]
        O-by-1 array representing the Strahler order for each arc in the flow graph
    vertex_ijs : NDArray[int]
        V'-by-2 array containing the ordered (i, j) incices of all arcs, concactinated together
    vertex_startends : NDArray[int]
        O-by-2 array containing the indices of where each arc starts and ends in `vertex_ijs`
        The returned endpoints are inclusive, meaning slicing must be done as `vertex_ijs[start : end + 1]`.
    """

    # Sort by arc order
    id = np.argsort(arc_orders)
    arc_orders = arc_orders[id]
    arc_endpts = arc_endpts[id, :]

    s_arc_orders = np.unique(arc_orders)
    s_arc_endpts = np.zeros((s_arc_orders.size, 2), dtype=np.int32)
    s_vertex_ijs = None

    for iarc in range(arc_orders.size):
        this_ijs = vertex_ijs[arc_endpts[iarc, 0] : arc_endpts[iarc, 1] + 1, :]
        if s_vertex_ijs is None:
            s_vertex_ijs = this_ijs
        else:
            s_vertex_ijs = np.concat(
                [s_vertex_ijs, np.array([[np.nan, np.nan]]), this_ijs], axis=0
            )
        if (iarc < arc_orders.size - 1) and (arc_orders[iarc] == arc_orders[iarc + 1]):
            continue
        s_arc_endpts[s_arc_orders == arc_orders[iarc], 1] = s_vertex_ijs.shape[0] - 1
        if arc_orders[iarc] < np.max(s_arc_orders):
            s_arc_endpts[s_arc_orders == arc_orders[iarc + 1], 0] = (
                s_vertex_ijs.shape[0] + 1
            )
    return s_arc_orders, s_vertex_ijs, s_arc_endpts


def _used_graph_vertices(
    ijs: npt.NDArray[np.number], endpts: npt.NDArray[np.integer]
) -> tuple[npt.NDArray[np.number], npt.NDArray[np.bool_]]:
    """
    Extracts the unique vertices referenced by the arcs of a graph.

    Parameters
    ----------
    ijs : NDArray[int | float]
        V-by-n array containing the coordinates of all stored vertices.
    endpts : NDArray[int]
        A-by-2 array containing the inclusive starting and ending vertex indices of each arc.

    Returns
    -------
    unique_ijs : NDArray[int | float]
        U-by-n array containing the unique coordinates referenced by at least one arc.
    unique_is_endpts : NDArray[bool]
        U-by-(1) boolean array indicating whether each unique coordinate is an endpoint of at least one arc.
    """
    endpts = np.asarray(endpts)

    # Expand inclusive endpoint ranges to exclude unreferenced entries in `ijs`
    used_ids = np.concatenate([np.arange(start, end + 1) for start, end in endpts])

    # Identify endpoints before coordinates shared by multiple arcs are merged
    endpoint_ids = np.concatenate((endpts[:, 0], endpts[:, 1]))
    is_endpoint = np.isin(used_ids, endpoint_ids)

    coords = ijs[used_ids]

    # Mark coordinates as endpoints when any occurrence is an endpoint
    unique_coords, inverse = np.unique(coords, axis=0, return_inverse=True)
    unique_is_endpoint = np.zeros(len(unique_coords), dtype=bool)
    np.logical_or.at(unique_is_endpoint, inverse, is_endpoint)

    return unique_coords, unique_is_endpoint


def find_graph_overlaps(
    g1_ijs: npt.NDArray[np.number],
    g1_endpts: npt.NDArray[np.integer],
    g2_ijs: npt.NDArray[np.number],
    g2_endpts: npt.NDArray[np.integer],
) -> tuple[
    npt.NDArray[np.number],
    npt.NDArray[np.number],
    npt.NDArray[np.number],
    npt.NDArray[np.number],
]:
    """
    Finds and classifies the shared vertices of two graphs.

    Only vertices referenced by the inclusive arc ranges in `g1_endpts` and `g2_endpts` are considered. Repeated coordinates within a graph are treated as endpoints when at least one occurrence is an endpoint.

    Parameters
    ----------
    g1_ijs : NDArray[int | float]
        V1-by-n array containing the coordinates of all vertices stored for the first graph.
    g1_endpts : NDArray[int]
        A1-by-2 array containing the inclusive starting and ending vertex indices of each arc in the first graph.
    g2_ijs : NDArray[int | float]
        V2-by-n array containing the coordinates of all vertices stored for the second graph.
    g2_endpts : NDArray[int]
        A2-by-2 array containing the inclusive starting and ending vertex indices of each arc in the second graph.

    Returns
    -------
    vert_vert_ijs : NDArray[int | float]
        Coordinates that are endpoints in both graphs.
    intr_intr_ijs : NDArray[int | float]
        Coordinates that are interior vertices in both graphs.
    g1_intr_g2_vert_ijs : NDArray[int | float]
        Coordinates that are interior vertices in the first graph and endpoints in the second graph.
    g1_vert_g2_intr_ijs : NDArray[int | float]
        Coordinates that are endpoints in the first graph and interior vertices in the second graph.
    """
    g1_coords, g1_is_endpts = _used_graph_vertices(g1_ijs, g1_endpts)
    g2_coords, g2_is_endpts = _used_graph_vertices(g2_ijs, g2_endpts)

    # Use a common type so the row keys are directly comparable
    dtype = np.result_type(g1_coords.dtype, g2_coords.dtype)
    g1_coords = np.ascontiguousarray(g1_coords, dtype=dtype)
    g2_coords = np.ascontiguousarray(g2_coords, dtype=dtype)

    # View each coordinate row as one value for a sparse set intersection
    row_dtype = np.dtype((np.void, dtype.itemsize * g1_coords.shape[1]))
    g1_keys = g1_coords.view(row_dtype).ravel()
    g2_keys = g2_coords.view(row_dtype).ravel()

    _, g1_ids, g2_ids = np.intersect1d(g1_keys, g2_keys, return_indices=True)

    overlaps: npt.NDArray[np.number] = g1_coords[g1_ids]
    g1_ep = g1_is_endpts[g1_ids]
    g2_ep = g2_is_endpts[g2_ids]

    # Partition the overlaps by their roles in the two graphs
    vert_vert = overlaps[g1_ep & g2_ep]
    intr_intr = overlaps[~g1_ep & ~g2_ep]
    g1_intr_g2_vert = overlaps[~g1_ep & g2_ep]
    g1_vert_g2_intr = overlaps[g1_ep & ~g2_ep]

    return (vert_vert, intr_intr, g1_intr_g2_vert, g1_vert_g2_intr)


def _find_overlap_neighbours(
    ijs: npt.NDArray[np.number],
    endpts: npt.NDArray[np.integer],
    overlaps: npt.NDArray[np.number],
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """Finds overlaps preceded or followed by another overlap within an arc."""
    prev_alsos = np.zeros(overlaps.shape[0], dtype=bool)
    after_alsos = np.zeros(overlaps.shape[0], dtype=bool)
    if overlaps.shape[0] == 0:
        return prev_alsos, after_alsos

    dtype = np.result_type(ijs.dtype, overlaps.dtype)
    overlap_coords = np.ascontiguousarray(overlaps, dtype=dtype)
    row_dtype = np.dtype((np.void, dtype.itemsize * overlaps.shape[1]))
    overlap_keys = overlap_coords.view(row_dtype).ravel()

    for start, end in endpts:
        arc_coords = np.ascontiguousarray(ijs[start : end + 1], dtype=dtype)
        arc_keys = arc_coords.view(row_dtype).ravel()
        overlap_ids = np.searchsorted(overlap_keys, arc_keys)
        matches = overlap_ids < overlap_keys.size
        matches[matches] &= overlap_keys[overlap_ids[matches]] == arc_keys[matches]

        # Record adjacency only between consecutive vertices of the same arc
        prev_matches = matches[1:] & matches[:-1]
        np.logical_or.at(prev_alsos, overlap_ids[1:][prev_matches], True)
        np.logical_or.at(after_alsos, overlap_ids[:-1][prev_matches], True)

    return prev_alsos, after_alsos


def solve_graph_overlaps(
    g1_orders: npt.NDArray[np.integer],
    g1_ijs: npt.NDArray[np.number],
    g1_endpts: npt.NDArray[np.integer],
    g2_orders: npt.NDArray[np.integer],
    g2_ijs: npt.NDArray[np.number],
    g2_endpts: npt.NDArray[np.integer],
    allows_arcs_overlap: bool = True,
) -> tuple[
    npt.NDArray[np.integer],
    npt.NDArray[np.number],
    npt.NDArray[np.integer],
    npt.NDArray[np.integer],
    npt.NDArray[np.number],
    npt.NDArray[np.integer],
]:
    """
    Splits two graphs at shared vertices to align their arc endpoints.

    Vertices that are endpoints in only one graph are inserted as endpoints in
    the other graph. Interior overlaps are inserted into both graphs unless
    they belong to a shared arc and `allows_arcs_overlap` is `True`.

    Parameters
    ----------
    g1_orders : NDArray[int]
        A1-by-(1) array containing the order of each arc in the first graph.
    g1_ijs : NDArray[int | float]
        V1-by-n array containing the coordinates of all vertices stored for the first graph.
    g1_endpts : NDArray[int]
        A1-by-2 array containing the inclusive starting and ending vertex indices of each arc in the first graph.
    g2_orders : NDArray[int]
        A2-by-(1) array containing the order of each arc in the second graph.
    g2_ijs : NDArray[int | float]
        V2-by-n array containing the coordinates of all vertices stored for the second graph.
    g2_endpts : NDArray[int]
        A2-by-2 array containing the inclusive starting and ending vertex indices of each arc in the second graph.
    allows_arcs_overlap : bool, optional
        Whether shared sequences of interior vertices may remain overlapping without being split into separate arcs.
        If true, consecutive overlap of vertices are isolated as a new arc, which will be identical between the two input graphs (aside form the directivity).
        The default options is `True`.

    Returns
    -------
    g1_orders : NDArray[int]
        Updated orders of the arcs in the first graph.
    g1_ijs : NDArray[int | float]
        Updated vertex coordinates of the first graph.
    g1_endpts : NDArray[int]
        Updated inclusive endpoint indices of the arcs in the first graph.
    g2_orders : NDArray[int]
        Updated orders of the arcs in the second graph.
    g2_ijs : NDArray[int | float]
        Updated vertex coordinates of the second graph.
    g2_endpts : NDArray[int]
        Updated inclusive endpoint indices of the arcs in the second graph.
    """
    _, overlaps, g1_intr_g2_vert, g1_vert_g2_intr = find_graph_overlaps(
        g1_ijs, g1_endpts, g2_ijs, g2_endpts
    )

    # Match endpoints already present in the first graph by splitting the second
    for new_endpt in g1_vert_g2_intr:
        g2_orders, g2_ijs, g2_endpts = insert_endpt(
            g2_orders, g2_ijs, g2_endpts, new_endpt
        )

    # Match endpoints already present in the second graph by splitting the first
    for new_endpt in g1_intr_g2_vert:
        g1_orders, g1_ijs, g1_endpts = insert_endpt(
            g1_orders, g1_ijs, g1_endpts, new_endpt
        )

    if allows_arcs_overlap:
        # Locate shared runs without assuming coordinates occur only once
        v_prev_alsos, v_after_alsos = _find_overlap_neighbours(
            g1_ijs, g1_endpts, overlaps
        )
        r_prev_alsos, r_after_alsos = _find_overlap_neighbours(
            g2_ijs, g2_endpts, overlaps
        )

        need_endpts = ~(v_prev_alsos & v_after_alsos & r_prev_alsos & r_after_alsos)

    # Split isolated crossings, or every interior overlap when arcs may not overlap
    for ivert, vert in enumerate(overlaps):
        if allows_arcs_overlap and (not need_endpts[ivert]):
            continue
        g1_orders, g1_ijs, g1_endpts = insert_endpt(g1_orders, g1_ijs, g1_endpts, vert)
        g2_orders, g2_ijs, g2_endpts = insert_endpt(g2_orders, g2_ijs, g2_endpts, vert)
    return g1_orders, g1_ijs, g1_endpts, g2_orders, g2_ijs, g2_endpts


def _convert_index_array_to_F_fmt(vertices: npt.NDArray) -> npt.NDArray:
    if vertices.shape[1] == 2 and vertices.shape[0] != 2:
        vertices = vertices
    elif vertices.shape[0] == 2 and vertices.shape[1] != 2:
        vertices = vertices.T
    elif vertices.shape == (2, 2):
        vertices = vertices
    else:
        raise ValueError("Array cannot be parsed as indices.")
    return vertices


def _simplify_multiple_flowgraphs(
    vertices_list: list[npt.NDArray[np.number]] | tuple[npt.NDArray[np.number], ...],
    endpts_list: list[npt.NDArray[np.integer]] | tuple[npt.NDArray[np.integer], ...],
    tol: int | float,
    check_topology: bool,
    backend: str,
) -> tuple[
    npt.NDArray[np.number]
    | list[npt.NDArray[np.number]]
    | tuple[npt.NDArray[np.number], ...],
    npt.NDArray[np.int32]
    | list[npt.NDArray[np.int32]]
    | tuple[npt.NDArray[np.int32], ...],
    npt.NDArray[np.bool_]
    | list[npt.NDArray[np.bool_]]
    | tuple[npt.NDArray[np.bool_], ...],
]:
    vertices_shps: list[tuple] = []
    endpts_shps: list[tuple] = []

    all_vertices_list: list[npt.NDArray[np.number]] = []
    all_endpts_list: list[npt.NDArray[np.integer]] = []
    all_graph_ids_list: list[npt.NDArray[np.uint8]] = []

    for i, (vertices, endpts) in enumerate(zip(vertices_list, endpts_list)):
        vertices_shps.append(vertices.shape)
        endpts_shps.append(endpts.shape)

        if vertices.ndim != 2 or endpts.ndim != 2:
            raise ValueError(
                f"Graph at index {i} has invalid dimension (must be 2D arrays)."
            )
        # Standardise vertices and endpoints arrays
        try:
            vertices = _convert_index_array_to_F_fmt(vertices)
        except ValueError:
            raise ValueError(
                f"Vertex array at index {i} shape {vertices.shape} is not V-by-2 or 2-by-V."
            )
        try:
            endpts = _convert_index_array_to_F_fmt(endpts)
        except ValueError:
            raise ValueError(
                f"Endpoint array at index {i} shape {endpts.shape} is not A-by-2 or 2-by-A."
            )

        all_vertices_list.append(vertices.copy())
        all_endpts_list.append(endpts.copy())

    # Insert endpoints at graph overlaps before simplifying any coordinates
    orders_list = [
        np.zeros(endpts.shape[0], dtype=np.uint8) for endpts in all_endpts_list
    ]
    for i in range(len(all_vertices_list) - 1):
        for j in range(i + 1, len(all_vertices_list)):
            (
                orders_list[i],
                all_vertices_list[i],
                all_endpts_list[i],
                orders_list[j],
                all_vertices_list[j],
                all_endpts_list[j],
            ) = solve_graph_overlaps(
                orders_list[i],
                all_vertices_list[i],
                all_endpts_list[i],
                orders_list[j],
                all_vertices_list[j],
                all_endpts_list[j],
                allows_arcs_overlap=True,
            )

    # Concatenate the graphs while retaining the graph membership of each arc
    offset: int = 0
    offset_endpts_list = []
    for i, (vertices, endpts) in enumerate(zip(all_vertices_list, all_endpts_list)):
        offset_endpts_list.append(endpts + offset)
        all_graph_ids_list.append(np.full(endpts.shape[0], i, dtype=np.uint8))
        offset += vertices.shape[0]
    all_vertices = np.concatenate(all_vertices_list, axis=0)
    all_endpts = np.concatenate(offset_endpts_list, axis=0)
    all_graph_ids = np.concatenate(all_graph_ids_list, axis=0)

    # Call the core single flowgraph simplifier
    _, _, keeps_concat = _simplify_single_flowgraph(
        all_vertices,
        all_endpts,
        tol=tol,
        check_topology=check_topology,
        backend=backend,
        graph_ids=all_graph_ids,
    )

    # Separate the simplified graph back into multiple graphs
    s_vertices_list = []
    s_endpts_list = []
    keeps_list = []

    offset = 0
    for i in range(len(all_vertices_list)):
        vertex_shp = vertices_shps[i]
        endpts_shp = endpts_shps[i]

        nvertices_i = all_vertices_list[i].shape[0]
        keeps_i = keeps_concat[offset : offset + nvertices_i]

        vertices = all_vertices_list[i]
        simp_v_i = vertices[keeps_i, :]

        vertex_cumsum_i = np.cumsum(keeps_i) - 1
        local_e_std = all_endpts_list[i]
        simp_e_i = vertex_cumsum_i[local_e_std]

        # Restore original orientation
        if vertex_shp[0] == 2 and vertex_shp[1] != 2:
            simp_v_i = simp_v_i.T
        if endpts_shp[0] == 2 and endpts_shp[1] != 2:
            simp_e_i = simp_e_i.T

        s_vertices_list.append(simp_v_i)
        s_endpts_list.append(simp_e_i)
        keeps_list.append(keeps_i)

        offset += nvertices_i

    if isinstance(vertices_list, tuple):
        return (
            tuple(s_vertices_list),
            tuple(s_endpts_list),
            tuple(keeps_list),
        )
    else:
        return s_vertices_list, s_endpts_list, keeps_list


def _ignore_identical_intergraph_arcs(
    intxs: Optional[npt.NDArray[np.int32]],
    vertices: npt.NDArray[np.number],
    endpts: npt.NDArray[np.integer],
    graph_ids: npt.NDArray[np.integer],
) -> Optional[npt.NDArray[np.int32]]:
    """
    Removes topology violations between identical arcs in different graphs.
    """
    if intxs is None:
        return None

    keeps = np.ones(intxs.shape[0], dtype=bool)
    identical_pairs: dict[tuple[int, int], bool] = {}
    for i, (iarc, jarc, _, _, _) in enumerate(intxs):
        if graph_ids[iarc] == graph_ids[jarc]:
            continue

        pair = (int(iarc), int(jarc))
        if pair not in identical_pairs:
            istart, iend = endpts[:, iarc]
            jstart, jend = endpts[:, jarc]
            iarc_vertices = vertices[:, istart : iend + 1]
            jarc_vertices = vertices[:, jstart : jend + 1]
            identical_pairs[pair] = np.array_equal(
                iarc_vertices, jarc_vertices
            ) or np.array_equal(iarc_vertices, jarc_vertices[:, ::-1])
        if identical_pairs[pair]:
            keeps[i] = False

    if not np.any(keeps):
        return None
    return intxs[keeps]


def _locate_disallowed_graph_topology(
    vertices: npt.NDArray[np.number],
    endpts: npt.NDArray[np.integer],
    graph_ids: Optional[npt.NDArray[np.integer]] = None,
) -> Optional[npt.NDArray[np.int32]]:
    """
    Locates violations in arrays stored in internal (2, N) layout.
    """
    intxs = locate_invalid_graph_topology(vertices.T, endpts.T)
    if graph_ids is not None:
        intxs = _ignore_identical_intergraph_arcs(intxs, vertices, endpts, graph_ids)
    return intxs


def _resolve_topology_intersections(
    vertices: npt.NDArray[np.number],
    endpts: npt.NDArray[np.integer],
    vertex_keeps: npt.NDArray[np.bool_],
    tol: float,
    graph_ids: Optional[npt.NDArray[np.integer]] = None,
    max_iters: int = 4,
) -> npt.NDArray[np.bool_]:
    """
    Checks for topology violations and resolves them by iteratively reducing
    the tolerance for conflicting arcs and re-simplifying them.
    """
    vertex_cumsum = np.cumsum(vertex_keeps) - 1
    vertices_aux = vertices[:, vertex_keeps]
    endpts_aux = vertex_cumsum[endpts]

    intxs = _locate_disallowed_graph_topology(vertices_aux, endpts_aux, graph_ids)

    niters = 0
    while (intxs is not None) and (niters <= max_iters):
        tol = float(tol) / 2
        niters += 1
        for iarc in np.unique(intxs[:, :2]):
            start = endpts[0, iarc]
            end = endpts[1, iarc]
            arc_length = end - start + 1
            vertex_keeps[start : end + 1] = graphs_f.simplify_flowgraph(
                vertices[:, start : end + 1].astype(np.float32, order="F"),
                np.array([[1], [arc_length]], dtype=np.int32, order="F"),
                tol,
            ).astype(bool)
        # Squeeze the vertices and map the arc endpoints to the new indices
        vertex_cumsum = np.cumsum(vertex_keeps) - 1
        vertices_aux = vertices[:, vertex_keeps]
        endpts_aux = vertex_cumsum[endpts]

        intxs = _locate_disallowed_graph_topology(vertices_aux, endpts_aux, graph_ids)
    # If there are still intersections after that many iterations, don't simplify those arc
    if intxs is not None:
        for iarc in np.unique(intxs[:, :2]):
            vertex_keeps[endpts[0, iarc] : endpts[1, iarc] + 1] = True
    return vertex_keeps.astype(bool)


def _simplify_single_flowgraph(
    vertices: npt.NDArray[np.number],
    endpts: npt.NDArray[np.integer],
    tol: float,
    check_topology: bool,
    backend: str,
    graph_ids: Optional[npt.NDArray[np.integer]] = None,
) -> tuple[npt.NDArray[np.number], npt.NDArray[np.integer], npt.NDArray[np.bool_]]:
    """
    Core function to simplify a single flow graph using RDP algorithm.
    """
    match backend:
        case "python":
            raise NotImplementedError(
                "The Python implementation of `simplify_flowgraph` is not implemented yet."
            )
        case "fortran":
            # Standardise inputs to FORTRAN layout (2, N) and (2, A)
            if not (vertices.shape[0] == 2 and vertices.shape[1] != 2):
                vertices = vertices.T
            if not (endpts.shape[0] == 2 and endpts.shape[1] != 2):
                endpts = endpts.T

            # Make a copy of arc_endpts to avoid modifying the input array in-place
            endpts = endpts.copy()

            # Convert 0-based Python indices to 1-based FORTRAN indices
            endpts += 1

            # Call the FORTRAN routine to get the boolean mask of kept vertices
            vertex_keeps = graphs_f.simplify_flowgraph(
                vertices.astype(np.float32, order="F"),
                endpts.astype(np.int32, order="F"),
                tol,
            ).astype(bool)

            # Revert back to 0-based Python indexing
            endpts -= 1

            if check_topology:
                vertex_keeps = _resolve_topology_intersections(
                    vertices, endpts, vertex_keeps, tol, graph_ids=graph_ids
                )

    # Squeeze the vertices and map the arc endpoints to the new indices
    vertex_cumsum = np.cumsum(vertex_keeps) - 1
    simp_vertices = vertices[:, vertex_keeps]
    simp_endpts = vertex_cumsum[endpts]

    if check_topology:
        final_intxs = _locate_disallowed_graph_topology(
            simp_vertices, simp_endpts, graph_ids
        )
        if final_intxs is not None:
            input_intxs = _locate_disallowed_graph_topology(vertices, endpts, graph_ids)
            if input_intxs is not None:
                raise InvalidOriginalGraphTopology(
                    "The simplified graph is invalid because the original input graph topology is invalid.",
                    input_intxs,
                )
            raise UnresolvedSimplificationTopology(
                "The final simplified graph has unresolved topology violations.",
                final_intxs,
            )

    # Transpose and cast arrays to C-contiguous layout for return
    simp_vertices = simp_vertices.T.astype(vertices.dtype, order="C")
    simp_endpts = simp_endpts.T.astype(np.int32, order="C")
    return simp_vertices, simp_endpts, vertex_keeps


@overload
def simplify_flowgraph(
    vertex_xys: npt.NDArray[np.number],
    arc_endpts: npt.NDArray[np.integer],
    tol: int | float = 1,
    check_topology: bool = True,
    backend: Literal["fortran", "python"] = "fortran",
) -> tuple[npt.NDArray[np.number], npt.NDArray[np.int32], npt.NDArray[np.bool_]]: ...


@overload
def simplify_flowgraph(
    vertex_xys: npt.NDArray[np.number],
    arc_endpts: npt.NDArray[np.integer],
    tol: int | float = 1,
    check_topology: bool = True,
    backend: Literal["fortran", "python"] = "fortran",
) -> tuple[npt.NDArray[np.number], npt.NDArray[np.int32], npt.NDArray[np.bool_]]: ...


@overload
def simplify_flowgraph(
    vertex_xys: list[npt.NDArray[np.number]],
    arc_endpts: list[npt.NDArray[np.integer]],
    tol: int | float = 1,
    check_topology: bool = True,
    backend: Literal["fortran", "python"] = "fortran",
) -> tuple[
    list[npt.NDArray[np.number]],
    list[npt.NDArray[np.int32]],
    list[npt.NDArray[np.bool_]],
]: ...


@overload
def simplify_flowgraph(
    vertex_xys: tuple[npt.NDArray[np.number]],
    arc_endpts: tuple[npt.NDArray[np.integer]],
    tol: int | float = 1,
    check_topology: bool = True,
    backend: Literal["fortran", "python"] = "fortran",
) -> tuple[
    tuple[npt.NDArray[np.number]],
    tuple[npt.NDArray[np.int32]],
    tuple[npt.NDArray[np.bool_]],
]: ...


def simplify_flowgraph(
    vertex_xys,
    arc_endpts,
    tol: int | float = 1,
    check_topology: bool = True,
    backend: Literal["fortran", "python"] = "fortran",
):
    """
    Simplify a flow graph using the Ramer-Douglas-Peucker (RDP) algorithm with a fixed tolerance threshold.

    When multiple graphs are supplied, their overlaps are first split into compatible arcs using `solve_graph_overlaps`. Identical arcs belonging to different graphs are ignored during topology validation, including when their vertex directions are reversed.

    Parameters
    ----------
    vertex_xys : NDArray[number] or Iterable[NDArray[number]]
        V-by-2 (or 2-by-V) array of coordinates representing the vertices in the flow graph, or an iterable of such arrays
    arc_endpts : NDArray[int] or Iterable[NDArray[int]]
        A-by-2 (or 2-by-A) array of indices indicating where each arc starts and ends in `vertex_xys`, or an iterable of such arrays
    tol : int | float, optional
        Tolerance threshold for simplification
        Vertices with perpendicular distance to the line segment less than or equal to `tol` will be simplified/removed.
        Default tolerance is 1.
    check_topology : bool, optional
        Whether to check for invalid topography in the simplified graph
        Default option is `True`.
    backend : {'fortran', 'python'}, optional
        Backend to use for computation
        Default backend and the only one currently available is `'fortran'`.

    Returns
    -------
    simp_vertex_xys : NDArray[number] or list/tuple of NDArray[number]
        V'-by-2 array of coordinates representing the simplified vertices, or a list/tuple of such arrays
    simp_arc_endpts : NDArray[int32] or list/tuple of NDArray[int32]
        A-by-2 array of indices indicating the start and end of each simplified arc, or a list/tuple of such arrays
    keeps : NDArray[bool] or list/tuple of NDArray[bool]
        V-by-1 mask indicating which of the input vertices are retained in the simplified graph, or a list/tuple of such masks.
        For multiple overlapping graphs, the masks refer to the intermediate vertex arrays produced by `solve_graph_overlaps`, which may contain additional vertices.

    Raises
    ------
    InvalidOriginalGraphTopology
        If the final result is invalid and the normalised input graph already
        contains disallowed topology violations.
    UnresolvedSimplificationTopology
        If the normalised input is valid but the final simplified graph
        contains disallowed topology violations.
    NotImplementedError
        If tries to call the not-yet-implemented Python backend.
    """

    if isinstance(vertex_xys, (list, tuple)) or isinstance(arc_endpts, (list, tuple)):
        if (not isinstance(vertex_xys, (list, tuple))) or (
            not isinstance(arc_endpts, (list, tuple))
        ):
            raise ValueError(
                "Arguments 'vertex_xys' and 'arc_endpts' must both be iterables (or neither)."
            )
        if len(vertex_xys) != len(arc_endpts):
            raise ValueError(
                f"Arguments 'vertex_xys' and 'arc_endpts' must have the same length, but got {len(vertex_xys)} and {len(arc_endpts)}, respectively, instead."
            )
        return _simplify_multiple_flowgraphs(
            vertex_xys,
            arc_endpts,
            tol=tol,
            check_topology=check_topology,
            backend=backend,
        )

    return _simplify_single_flowgraph(
        vertex_xys, arc_endpts, tol=tol, check_topology=check_topology, backend=backend
    )


def _raise_topology_scan_error(err_code: int) -> None:
    """Translates a FORTRAN topology-scanner status into a Python exception.

    Parameters
    ----------
    err_code : int
        Scanner status code. Zero indicates success, one invalid inputs, and
        two a workspace-allocation failure.

    Raises
    ------
    ValueError
        If the scanner rejected its array shapes or output capacity.
    MemoryError
        If the scanner could not allocate its internal workspace.
    RuntimeError
        If the scanner returned an unknown nonzero status.
    """
    if err_code == 1:
        raise ValueError("Invalid array shapes or output capacity passed.")
    elif err_code == 2:
        raise MemoryError("Unable to allocate topology-intersection workspace.")
    elif err_code != 0:
        raise RuntimeError(
            f"Unexpected topology-intersection scanner error code: {err_code}."
        )


def _locate_invalid_graph_topology_fortran(
    vertex_xys: npt.NDArray[np.number],
    arc_endpts: npt.NDArray[np.integer],
) -> Optional[npt.NDArray[np.int32]]:
    """
    Returns every topology violation using the capacity-aware FORTRAN scanner.

    The first scan uses a small provisional output buffer. If the exact count
    reported by that scan exceeds the buffer, the scan is repeated with a
    buffer of exactly the required size. Incomplete provisional results are
    never returned.

    Parameters
    ----------
    vertex_xys : NDArray[number]
        Vertex coordinates with shape `(nvertices, 2)`.
    arc_endpts : NDArray[integer]
        Inclusive, zero-based arc endpoint indices with shape `(narcs, 2)`.

    Returns
    -------
    NDArray[int32] or None
        Complete `(nintxs, 5)` intersection records using zero-based indices,
        or `None` when no violations are found.

    Raises
    ------
    ValueError
        If the low-level scanner rejects its inputs.
    MemoryError
        If scanner workspace or result allocation fails.
    RuntimeError
        If the scanner returns an unexpected status or the exact count changes
        during the retry.
    """
    vertices_f = np.asfortranarray(vertex_xys.T, dtype=np.float32)
    endpts_f = np.asfortranarray(arc_endpts.T, dtype=np.int32) + 1
    capacity = max(vertices_f.shape[1] // 100, 3) # Arbitrary capacity that seems to work

    intxs, nintxs, err_code = graphs_f.scan_invalid_graph_topology(
        vertices_f, endpts_f, capacity
    )
    _raise_topology_scan_error(err_code)

    if nintxs == 0:
        return None

    if nintxs > capacity:
        expected_nintxs = nintxs
        intxs, nintxs, err_code = graphs_f.scan_invalid_graph_topology(
            vertices_f, endpts_f, expected_nintxs
        )
        _raise_topology_scan_error(err_code)
        if nintxs != expected_nintxs:
            raise RuntimeError(
                "Topology-intersection count changed during exact-size retry."
            )

    intxs = intxs[:, :nintxs]
    intxs[:-1, :] -= 1  # Convert to 0-based indexing, except the intersection flag
    return intxs.T.astype(np.int32, order="C")


def locate_invalid_graph_topology(
    vertex_xys: npt.NDArray[np.number],
    arc_endpts: npt.NDArray[np.integer],
    backend: Literal["fortran", "python"] = "fortran",
) -> Optional[npt.NDArray[np.int32]]:
    """
    Locates invalid topologies (segment intersections) within and between arcs in a graph.

    This function checks for self-intersections within individual arcs, as well as intersections between segments of different arcs. The intersection checks are performed using a 2D line segment intersection algorithm.

    Parameters
    ----------
    vertex_xys : NDArray[number]
        2D array of shape `(nvertices, 2)` representing the grid coordinates (i, j) of each vertex.
    arc_endpts : NDArray[integer]
        2D array of shape `(narcs, 2)` containing the start and end vertex indices for each arc in `vertex_ijs`.
    backend : {'fortran', 'python'}, optional
        Computational backend to use.
        The default option is `'fortran'`.

    Returns
    -------
    NDArray[int32] or None
        2D array of shape `(nintxs, 5)` representing the detected intersections, or `None` if no intersections are found.
        The rows are sorted lexicographically and each row contains:
        - `iarc`: Index of the first arc (0-based).
        - `jarc`: Index of the second arc (0-based).
        - `iseg`: Start vertex index of the first intersecting segment (0-based).
        - `jseg`: Start vertex index of the second intersecting segment (0-based).
        - `intx_flag`: Flag indicating the type of intersection:
            - 1 : Interior-interior crossing (X).
            - 2 : Collinear overlap, not identical.
            - 3 : Identical segment.
            - 4 : Endpoint-on-interior (T-junction).
            - 5 : Degenerate segment (some line is actually a point).

    Raises
    ------
    ValueError
        If the shape of `vertex_ijs` or `arc_endpts` is invalid.
    MemoryError
        If the FORTRAN backend cannot allocate its scan workspace or result.
    RuntimeError
        If the FORTRAN scanner returns an unexpected error or an inconsistent
        count during the exact-size retry.
    """
    if vertex_xys.ndim != 2 or vertex_xys.shape[1] != 2:
        raise ValueError("Invalid array shapes passed.")
    elif arc_endpts.ndim != 2 or arc_endpts.shape[1] != 2:
        raise ValueError("Invalid array shapes passed.")

    match backend:
        case "python":
            from .graphs_py import _locate_invalid_graph_topology_py

            intxs = _locate_invalid_graph_topology_py(
                arc_endpts.astype(np.int32, order="C"),
                vertex_xys.astype(np.float64, order="C"),
            )
            if not intxs:
                return None
            intxs = np.array(intxs, dtype=np.int32, order="C")
        case "fortran":
            intxs = _locate_invalid_graph_topology_fortran(vertex_xys, arc_endpts)
            if intxs is None:
                return None
    if intxs.shape[0] > 1:
        # Sort lexicographically
        sort_idx = np.lexsort((intxs[:, 3], intxs[:, 2], intxs[:, 1], intxs[:, 0]))
        intxs = intxs[sort_idx]
    return intxs
