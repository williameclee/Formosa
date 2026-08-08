# Last modified
#   2026-07-01, En-Chi Lee (williameclee@gmail.com)
#     - Opted out of the out-of-bound check in `compute_downstream_indices` in `create_flowgraph`
#     - Added function `construct_flowgraph`
#   2026-07-09, En-Chi Lee (williameclee@gmail.com)
#     - Specified endpoint index definition for `construct_flowgraph`
#     - Implemented FORTRAN backend of function `simplify_flowgraph` and function `concat_flowgraph`
#     - Added vertex mask to output of function `simplify_flowgraph`
#   2026-07-12, En-Chi Lee (williameclee@gmail.com)
#     - Implemented Python and FORTRAN backends of function `locate_invalid_graph_topology`
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
#     - Various minor refactors and type annotation enhancements
#   2026-07-31, En-Chi Lee (williameclee@gmail.com)
#     - Preserved arc orders when `simplify_flowgraph` splits overlapping graphs
#   2026-08-03, En-Chi Lee (williameclee@gmail.com)
#     - Made `solve_graph_overlaps` stable and recognised already valid shared arcs
#   2026-08-04, En-Chi Lee (williameclee@gmail.com)
#     - Made `simplify_flowgraph` able to handle empty graphs
#     - Accelerated graph validation and simplification
#     - Implemented function `remove_unused_vertices`


from dataclasses import dataclass
import numpy as np

from formosa.geomorphology.flowdir.directions import D8Directions
from formosa.geomorphology.flowdir.utils import (
    compute_downstream_indices,
    raise_fortran_error,
)
import formosa.geomorphology.flowdir.flowdir as flowdir_m
import formosa.geomorphology.flowdir.raster as raster_m
from formosa.geomorphology.flowdir_f import flowdir_graphs as graphs_f

import warnings

import numpy.typing as npt
from typing import Literal, Iterable, Optional, TypeVar, overload

NpIndex = TypeVar("NpIndex", np.int32, np.int64, np.intp)


class GraphTopologyError(RuntimeError):
    """
    Base exception for a graph that fails topology validation.
    """


class InvalidOriginalGraphTopology(GraphTopologyError):
    """
    Raised when an invalid result originated from invalid input topology.
    """


class UnresolvedSimplificationTopology(GraphTopologyError):
    """
    Raised when simplification leaves invalid topology from valid input.
    """


class DirectedFlowCycleError(GraphTopologyError):
    """
    Raised when the selected flow field contains one or more directed cycles.
    """

    def __init__(self, cycle_ijs: npt.NDArray[np.integer]) -> None:
        self.cycle_ijs = np.asarray(cycle_ijs, dtype=np.int32).copy()
        super().__init__(
            "Selected flow graph contains directed cycles at "
            f"{self.cycle_ijs.tolist()}."
        )


class IncompleteFlowGraphError(GraphTopologyError):
    """
    Raised when construction omits one or more selected directed edges.
    """

    def __init__(
        self,
        missing_ijs: npt.NDArray[np.integer],
        missing_edges: Optional[npt.NDArray[np.integer]] = None,
    ) -> None:
        self.missing_ijs = np.asarray(missing_ijs, dtype=np.int32).copy()
        if missing_edges is None:
            self.missing_edges = np.empty((0, 4), dtype=np.int32)
        else:
            self.missing_edges = np.asarray(missing_edges, dtype=np.int32).copy()
        super().__init__(
            "Flow-graph construction omitted selected directed edges "
            f"{self.missing_edges.tolist()}; participating cells are "
            f"{self.missing_ijs.tolist()}."
        )


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
        2D array representing the flow directions for each cell.
    valids : NDArray[bool], optional
        Boolean mask array indicating valid cells in the flow direction grid.
        If `None`, all cells are considered valid.
        Default is `None`.
    directions : D8Directions, optional
        Instance of `D8Directions` defining the flow direction scheme.
        Default is `D8Directions()`.
    x : NDArray[number], optional
        2D array representing the x-coordinates of each cell.
        If provided, the graph will use these coordinates instead of grid indices.
        Default is `None`.
    y : NDArray[number], optional
        2D array representing the y-coordinates of each cell.
        If provided, the graph will use these coordinates instead of grid indices.
        Default is `None`.

    Returns
    -------
    graphi : NDArray[int]
        1D array representing the row indices of the graph edges.
    graphj : NDArray[int]
        1D array representing the column indices of the graph edges.
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
        dirs, dir_scheme=dir_scheme, check=False, return_flat_index=False
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


def _valid_flow_edges(
    dirs: npt.NDArray[np.integer],
    valids: npt.NDArray[np.bool_],
    dir_scheme: D8Directions,
) -> tuple[
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.bool_],
]:
    """
    Returns downstream indices and a mask indicating whether the cell flows into a valid neighbouring (non-self) edge.
    """
    dsi, dsj, _, ds_inbounds = compute_downstream_indices(
        dirs,
        dir_scheme=dir_scheme,
        check=False,
        return_flat_index=False,
        oob_is_okay=True,
    )

    # Whether the downstream cell is also valid (not just inbound)
    ds_valids = np.zeros(dirs.shape, dtype=bool)
    ds_valids[ds_inbounds] = valids[dsi[ds_inbounds], dsj[ds_inbounds]]

    # Exclude self-loops (where offsets di, dj == 0)
    not_self = dirs != dir_scheme.no_flow_code

    has_valid_ds = valids & ds_valids & not_self
    return dsi, dsj, has_valid_ds


def _validate_flowgraph_coverage(
    vertex_ijs: npt.NDArray[np.integer],
    arc_endpts: npt.NDArray[np.integer],
    dsi: npt.NDArray[np.integer],
    dsj: npt.NDArray[np.integer],
    has_valid_ds: npt.NDArray[np.bool_],
) -> None:
    """
    Checks that every selected directed edge occurs in a returned graph arc.
    """
    represented = np.zeros(has_valid_ds.shape, dtype=bool)

    # Identify consecutive vertex pairs that belong to an arc.
    segment_counts = np.zeros(vertex_ijs.shape[0], dtype=np.int32)
    np.add.at(segment_counts, arc_endpts[:, 0], 1)
    np.add.at(segment_counts, arc_endpts[:, 1], -1)
    segment_valids = np.cumsum(segment_counts)[:-1] > 0

    sources = vertex_ijs[:-1][segment_valids]
    targets = vertex_ijs[1:][segment_valids]

    # Confirm each represented edge matches the source cell's expected downstream.
    matches = (targets[:, 0] == dsi[sources[:, 0], sources[:, 1]]) & (
        targets[:, 1] == dsj[sources[:, 0], sources[:, 1]]
    )
    matched_sources = sources[matches]
    represented[matched_sources[:, 0], matched_sources[:, 1]] = True

    missing_sources = np.argwhere(has_valid_ds & ~represented)
    if missing_sources.size:
        missing_targets = np.column_stack(
            (
                dsi[missing_sources[:, 0], missing_sources[:, 1]],
                dsj[missing_sources[:, 0], missing_sources[:, 1]],
            )
        )
        missing_edges = np.column_stack((missing_sources, missing_targets))
        missing_ijs = np.unique(missing_edges.reshape(-1, 2), axis=0)
        raise IncompleteFlowGraphError(missing_ijs, missing_edges)


def construct_flowgraph(
    dirs: npt.NDArray[np.integer],
    dir_scheme: D8Directions = D8Directions(),
    valids: Optional[npt.NDArray[np.bool_]] = None,
    min_order: int = 2,
    orders: Optional[npt.NDArray[np.integer]] = None,
    preserve_junctions: bool = True,
    sort: bool = True,
    backend: Literal["fortran", "python"] = "fortran",
    remove_unused: bool = False,
) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """
    Constructs a flow graph from a flow direction grid.

    Parameters
    ----------
    dirs : NDArray[int], optional
        2D array representing the flow directions for each cell.
    dir_scheme : D8Directions, optional
        Instance of `D8Directions` defining the flow direction scheme.
        Default scheme is `D8Directions()`.
    valids : NDArray[bool], optional
        Boolean mask array indicating valid cells in the flow direction grid.
        If `None`, all cells are considered valid.
        Default mask is `None`.
    min_order : int, optional
        Minimum Strahler order to include in the flow graph (see `orders`).
        Default order is 2.
    orders : NDArray[uint8], optional
        2D integer array representing the Strahler order for each cell.
        If `None`, it will be computed from the flow direction grid.
        Default input is `None`.
    preserve_junctions : bool, optional
        Whether to preserve junctions in the flow graph.
        Default option is `True`.
    sort : bool, option
        Whether to sort the flow graph by arc order and then by length.
        Default option is `True`.
    backend : {'fortran', 'python'}, optional
        The backend to use for computation.
        `'fortran'` uses the FORTRAN extension for performance, while 'python' uses a pure Python implementation.
        Default backend is `'fortran'`.
    remove_unused : bool, optional
        Whether to compact the vertex array after construction so the arc ranges are adjacent in arc order.
        Default option is `False`.

    Returns
    -------
    arc_orders : NDArray[int8]
        1D array representing the Strahler order for each arc in the flow graph.
    vertex_ijs : NDArray[int32]
        V-by-2 array containing the ordered (i, j) incices of all arcs, concactinated together.
    vertex_endpts : NDArray[int32]
        A-by-2 array containing the indices of where each arc starts and ends in `vertex_ijs`.
        The returned endpoints are inclusive, meaning slicing must be done as `vertex_ijs[start : end + 1]`.

    Raises
    ------
    DirectedFlowCycleError
        If the selected flow field contains a directed cycle.
    IncompleteFlowGraphError
        If either endpoint of a selected directed edge is absent from the
        constructed graph.

    Notes
    -----
    Selected cells with no selected incoming or outgoing edge are intentionally
    omitted from the arc representation.
    """
    if valids is None:
        valids = np.ones(dirs.shape, dtype=bool)
    if orders is None:
        orders = raster_m.compute_flow_strahler_order(
            dirs,
            dir_scheme=dir_scheme,
            valids=valids,
            backend=backend,
        )

    # Find seed cells to start with
    valids = valids & (orders >= min_order)
    ncells = int(np.sum(valids))
    indegs = flowdir_m.count_indegree(
        dirs, dir_scheme=dir_scheme, valids=valids, backend=backend
    )
    cyclics = flowdir_m.find_cyclic_flowdirs(
        dirs,
        dir_scheme=dir_scheme,
        valids=valids,
        indegs=indegs,
        backend=backend,
    )
    cycle_ijs = np.argwhere(cyclics).astype(np.int32, order="C")
    if cycle_ijs.size > 0:
        raise DirectedFlowCycleError(cycle_ijs)

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
            narcs, nvertices, arc_orders, vertex_ijs, arc_endpts, err_code = (
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
            raise_fortran_error("construct_flowgraph", err_code)
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

    dsi, dsj, has_valid_ds = _valid_flow_edges(dirs, valids, dir_scheme)
    _validate_flowgraph_coverage(vertex_ijs, arc_endpts, dsi, dsj, has_valid_ds)

    if remove_unused:
        vertex_ijs, arc_endpts = remove_unused_vertices(vertex_ijs, arc_endpts)  # type: ignore

    return (
        arc_orders,
        vertex_ijs.astype(np.int32, order="C"),
        arc_endpts.astype(np.int32, order="C"),
    )


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
    vert : NDArray[int | float]
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
) -> Optional[int]:
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
        Default option is `True`.

    Returns
    -------
    iarc : int | None
        Index of the arc that contains the vertex of a given index, or `None` if the vertex is not a part of any arc.
    """
    ...


@overload
def find_arc_id_of_vertex(
    endpts: npt.NDArray[np.integer], ivert: Iterable[int], is_inclusive: bool = True
) -> Optional[list[int]]:
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
        Default option is `True`.

    Returns
    -------
    iarc : list[int] | None
        Indices of the arcs that contain the vertices of the given indices, or `None` if the vertices are not a part of any arc.
    """
    ...


def find_arc_id_of_vertex(
    endpts: npt.NDArray[np.integer],
    ivert: int | Iterable[int],
    is_inclusive: bool = True,
) -> Optional[int | list[int]]:
    def _find_arc_of_vertex(
        endpts: npt.NDArray[np.integer], ivert: int, is_inclusive: bool = True
    ) -> Optional[int]:
        iarc = np.flatnonzero(
            (ivert >= endpts[:, 0]) & (ivert <= (endpts[:, 1] - (not is_inclusive)))
        )
        if np.size(iarc) == 0:
            warnings.warn("Provided vertex is not a part of any arc.")
            return None
        elif np.size(iarc) > 1:
            raise ValueError("Provided vertex is found in multiple arcs.")
        return iarc[0]

    if isinstance(ivert, int) or (np.size(ivert) == 1):  # type: ignore
        iarc = _find_arc_of_vertex(endpts, ivert, is_inclusive)  # type: ignore
        return iarc
    iarc = [_find_arc_of_vertex(endpts, ivert, is_inclusive) for ivert in ivert]
    iarc = [iarc_ for iarc_ in iarc if iarc_ is not None]  # Reduce the list
    return iarc


def insert_endpt(
    orders: npt.NDArray[np.integer],
    ijs: npt.NDArray[np.number],
    endpts: npt.NDArray[np.integer],
    add_endpt: npt.NDArray[np.number] | int,
    remove_unused: bool = False,
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
    remove_unused : bool, optional
        Whether to compact the returned vertex array so the arc ranges are adjacent.
        Default option is `False`.

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

    def _return_graph(orders, ijs, endpts):
        if remove_unused:
            ijs, endpts = remove_unused_vertices(ijs, endpts)
        return orders, ijs, endpts

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
            return _return_graph(orders, ijs, endpts)

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
            return _return_graph(orders, ijs, endpts)
        ivert: int | list[int] = int(iverts[0]) if iverts.size == 1 else iverts.tolist()
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
        return _return_graph(orders, ijs, endpts)

    assert isinstance(iarc, list)  # Just for static type checking
    for jvert, jarc in zip(ivert, iarc):
        if jarc is None:
            continue
        orders, ijs, endpts = _insert_endpt(orders, ijs, endpts, jarc, jvert)
    return _return_graph(orders, ijs, endpts)


def concat_flowgraph(
    arc_orders: npt.NDArray[np.integer],
    vertex_ijs: npt.NDArray[np.integer],
    arc_endpts: npt.NDArray[np.integer],
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer], npt.NDArray[np.integer]]:
    """
    Concatenates arcs of the same order in a flow graph, separated by NaNs.
    It mainly serves to reduce the number of drawing calls when visualising the graph.

    Parameters
    ----------
    arc_orders : NDArray[int]
        O-by-1 array representing the Strahler order for each arc in the flow graph.
    vertex_ijs : NDArray[int]
        V-by-2 array containing the ordered (i, j) incices of all arcs, concactinated together.
    vertex_startends : NDArray[int]
        A-by-2 array containing the indices of where each arc starts and ends in `vertex_ijs`.
        The returned endpoints are inclusive, meaning slicing must be done as `vertex_ijs[start : end + 1]`.

    Returns
    ----------
    arc_orders : NDArray[int]
        O-by-1 array representing the Strahler order for each arc in the flow graph.
    vertex_ijs : NDArray[int]
        V'-by-2 array containing the ordered (i, j) incices of all arcs, concactinated together.
    vertex_startends : NDArray[int]
        O-by-2 array containing the indices of where each arc starts and ends in `vertex_ijs`.
        The returned endpoints are inclusive, meaning slicing must be done as `vertex_ijs[start : end + 1]`.
    """
    # Input validation
    assert np.size(arc_orders, 0) == np.size(arc_endpts, 0), (
        "The order and endpoint arrays must have the same length, "
        + f"but got {np.size(arc_orders, 0)} and {np.size(arc_endpts, 0)}, respectively, instead"
    )
    if np.size(arc_orders, 0) == 0:
        return arc_orders, vertex_ijs, arc_endpts

    # Sort by arc order
    id = np.argsort(arc_orders)
    arc_orders = arc_orders[id]
    arc_endpts = arc_endpts[id, :]

    s_arc_orders, first_group_ids = np.unique(arc_orders, return_index=True)
    arc_lengths = arc_endpts[:, 1] - arc_endpts[:, 0] + 1
    output_size = int(np.sum(arc_lengths) + arc_orders.size - 1)
    output_dtype = (
        vertex_ijs.dtype
        if arc_orders.size == 1
        else np.result_type(vertex_ijs.dtype, np.float64)
    )
    s_vertex_ijs = np.full(
        (output_size, vertex_ijs.shape[1]), np.nan, dtype=output_dtype
    )
    s_arc_endpts = np.zeros((s_arc_orders.size, 2), dtype=np.int32)

    cursor = 0
    group_id = 0
    for iarc, (start, end) in enumerate(arc_endpts):
        if iarc in first_group_ids:
            group_id = int(np.searchsorted(first_group_ids, iarc))
            s_arc_endpts[group_id, 0] = cursor
        length = int(end - start + 1)
        s_vertex_ijs[cursor : cursor + length] = vertex_ijs[start : end + 1]
        cursor += length
        s_arc_endpts[group_id, 1] = cursor - 1
        if iarc < arc_orders.size - 1:
            cursor += 1

    return s_arc_orders, s_vertex_ijs, s_arc_endpts


def remove_unused_vertices(
    vertices: npt.NDArray[np.number],
    endpts: npt.NDArray[NpIndex],
) -> tuple[npt.NDArray[np.number], npt.NDArray[NpIndex]]:
    """
    Removes stored vertices that are not referenced by any graph arc.

    Arcs retain their input order and their vertices are copied into adjacent
    ranges. Consequently, the start of every arc after the first is one index
    beyond the end of the preceding arc. Arc endpoint indices are inclusive.

    Parameters
    ----------
    vertices : NDArray[int | float]
        V-by-n array of stored vertex coordinates.
    endpts : NDArray[int]
        A-by-2 array of inclusive arc ranges into `vertices`.

    Returns
    -------
    vertices : NDArray[int | float]
        Compact vertex array containing only vertices referenced by arcs.
    endpts : NDArray[int]
        Arc ranges remapped into the compact vertex array.

    Raises
    ------
    ValueError
        If the input arguments have the wrong shapes.
    """
    vertices = np.asarray(vertices)
    endpts = np.asarray(endpts)

    if vertices.ndim != 2:
        raise ValueError("vertices must be a two-dimensional array.")
    if endpts.ndim != 2 or endpts.shape[1] != 2:
        raise ValueError("endpts must have shape (number of arcs, 2).")
    if endpts.shape[0] == 0:
        return vertices[:0].copy(), endpts.copy()
    if np.any(endpts[:, 0] < 0) or np.any(endpts[:, 1] < endpts[:, 0]):
        raise ValueError(
            "Each arc must have a non-negative start no greater than its end."
        )
    if np.any(endpts[:, 1] >= vertices.shape[0]):
        raise ValueError("Arc endpoints must index rows in vertices.")

    lengths = endpts[:, 1] - endpts[:, 0] + 1
    compact_vertices = np.concatenate(
        [vertices[start : end + 1] for start, end in endpts], axis=0
    )
    compact_ends = np.cumsum(lengths, dtype=np.intp) - 1
    compact_starts = np.concatenate(([0], compact_ends[:-1] + 1))
    compact_endpts = np.column_stack((compact_starts, compact_ends)).astype(
        endpts.dtype, copy=False
    )
    return compact_vertices, compact_endpts


@dataclass
class _GraphVertexAnalysis:
    """
    Reusable vertex metadata for one flow graph.

    Attributes
    ----------
    verts : NDArray[int | float]
        Unique vertex coordinates referenced by at least one graph arc.
    is_endpt : NDArray[bool]
        Whether any occurrence of each vertex in `verts` is an arc endpoint.
    vert_cnts : NDArray[int]
        Number of occurrences of each vertex within the graph's arc ranges.
    vert_inv_ids : NDArray[int]
        Maps each row of the original vertex array to its row in `verts`.
        Stored vertices that are not referenced by an arc are assigned `-1`.
    """

    verts: npt.NDArray[np.number]
    is_endpt: npt.NDArray[np.bool_]
    vert_cnts: npt.NDArray[np.intp]
    vert_inv_ids: npt.NDArray[np.intp]


@dataclass
class _SharedVertexAnalysis:
    """
    Vertex metadata shared by a pair of flow graphs.

    Attributes
    ----------
    verts : NDArray[int | float]
        Unique vertex coordinates referenced by both graphs.
    g1_vert_ids : NDArray[int]
        Maps each row in `verts` to its unique-vertex ID in the first graph.
    g2_vert_ids : NDArray[int]
        Maps each row in `verts` to its unique-vertex ID in the second graph.
    g1_is_endpt : NDArray[bool]
        Whether each shared vertex is an endpoint in the first graph.
    g2_is_endpt : NDArray[bool]
        Whether each shared vertex is an endpoint in the second graph.
    g1_cnts : NDArray[int]
        Number of occurrences of each shared vertex in the first graph.
    g2_cnts : NDArray[int]
        Number of occurrences of each shared vertex in the second graph.
    """

    verts: npt.NDArray[np.number]
    g1_vert_ids: npt.NDArray[np.intp]
    g2_vert_ids: npt.NDArray[np.intp]
    g1_is_endpt: npt.NDArray[np.bool_]
    g2_is_endpt: npt.NDArray[np.bool_]
    g1_cnts: npt.NDArray[np.intp]
    g2_cnts: npt.NDArray[np.intp]


def _analyse_graph_vertices(
    verts: npt.NDArray[np.number], endpts: npt.NDArray[np.integer]
) -> _GraphVertexAnalysis:
    """
    Builds reusable vertex IDs, endpoint roles, and occurrence counts.

    Only rows referenced by an inclusive range in `endpts` contribute to the unique vertices, roles, and counts. The inverse-ID array retains the shape of the original stored vertex array so later operations can work directly with arc endpoint indices.

    Parameters
    ----------
    verts : NDArray[int | float]
        V-by-n array of stored vertex coordinates.
    endpts : NDArray[int]
        A-by-2 array of inclusive arc ranges into `verts`.

    Returns
    -------
    _GraphVertexAnalysis
        Unique referenced vertices and their reusable graph metadata.
    """
    endpts = np.asarray(endpts)
    nverts = verts.shape[0]
    if endpts.shape[0] == 0:
        return _GraphVertexAnalysis(
            verts=np.empty((0, verts.shape[1]), dtype=verts.dtype),
            is_endpt=np.empty((0,), dtype=bool),
            vert_cnts=np.empty((0,), dtype=np.intp),
            vert_inv_ids=np.full(verts.shape[0], -1, dtype=np.intp),
        )

    # Expand the inclusive arc ranges, excluding stored rows unused by any arc
    used_ids: npt.NDArray[np.integer] = np.concatenate(
        [np.arange(start, end + 1) for start, end in endpts]
    )
    endpt_ids = np.concatenate((endpts[:, 0], endpts[:, 1]))
    used_is_endpt = np.isin(used_ids, endpt_ids)

    # `vert_inv` maps each used occurrence to its deduplicated vertex
    verts, vert_inv, vert_cnts = np.unique(
        verts[used_ids], axis=0, return_inverse=True, return_counts=True
    )
    # A repeated coordinate is an endpoint if any of its occurrences is one
    is_endpt = np.zeros(verts.shape[0], dtype=bool)
    np.logical_or.at(is_endpt, vert_inv, used_is_endpt)

    # Restore a lookup indexed like the original stored vertex array
    vert_inv_ids = np.full(nverts, -1, dtype=np.intp)
    vert_inv_ids[used_ids] = vert_inv
    return _GraphVertexAnalysis(
        verts=verts,
        is_endpt=is_endpt,
        vert_cnts=vert_cnts.astype(np.intp, copy=False),
        vert_inv_ids=vert_inv_ids,
    )


def _find_shared_graph_vertices(
    g1: _GraphVertexAnalysis, g2: _GraphVertexAnalysis
) -> _SharedVertexAnalysis:
    """
    Intersects two reusable graph-vertex analyses.

    Parameters
    ----------
    g1 : _GraphVertexAnalysis
        Unique-vertex metadata for the first graph.
    g2 : _GraphVertexAnalysis
        Unique-vertex metadata for the second graph.

    Returns
    -------
    _SharedVertexAnalysis
        Shared vertices, their unique-vertex IDs in both graphs, endpoint roles, and occurrence counts.

    Notes
    -----
    Vertex coordinate rows are viewed as fixed-width byte values so NumPy can perform a 1D set intersection without Python tuple conversion.
    """
    dtype = np.result_type(g1.verts.dtype, g2.verts.dtype)
    g1_verts = np.ascontiguousarray(g1.verts, dtype=dtype)
    g2_verts = np.ascontiguousarray(g2.verts, dtype=dtype)
    # Encode each vertex row as one fixed-width value for `intersect1d`.
    row_dtype = np.dtype((np.void, dtype.itemsize * g1_verts.shape[1]))
    g1_keys = g1_verts.view(row_dtype).ravel()
    g2_keys = g2_verts.view(row_dtype).ravel()
    _, g1_ids, g2_ids = np.intersect1d(g1_keys, g2_keys, return_indices=True)
    return _SharedVertexAnalysis(
        verts=g1_verts[g1_ids],
        g1_vert_ids=g1_ids.astype(np.intp, copy=False),
        g2_vert_ids=g2_ids.astype(np.intp, copy=False),
        g1_is_endpt=g1.is_endpt[g1_ids],
        g2_is_endpt=g2.is_endpt[g2_ids],
        g1_cnts=g1.vert_cnts[g1_ids],
        g2_cnts=g2.vert_cnts[g2_ids],
    )


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
    endpt_endpt_ijs : NDArray[int | float]
        Coordinates that are endpoints in both graphs.
    intr_intr_ijs : NDArray[int | float]
        Coordinates that are interior vertices in both graphs.
    g1_intr_g2_endpt_ijs : NDArray[int | float]
        Coordinates that are interior vertices in the first graph and endpoints in the second graph.
    g1_endpt_g2_intr_ijs : NDArray[int | float]
        Coordinates that are endpoints in the first graph and interior vertices in the second graph.
    """
    shared = _find_shared_graph_vertices(
        _analyse_graph_vertices(g1_ijs, g1_endpts),
        _analyse_graph_vertices(g2_ijs, g2_endpts),
    )
    overlaps = shared.verts

    # Partition the overlaps by their roles in the two graphs
    endpt_endpt = overlaps[shared.g1_is_endpt & shared.g2_is_endpt]
    intr_intr = overlaps[~shared.g1_is_endpt & ~shared.g2_is_endpt]
    g1_intr_g2_endpt = overlaps[~shared.g1_is_endpt & shared.g2_is_endpt]
    g1_endpt_g2_intr = overlaps[shared.g1_is_endpt & ~shared.g2_is_endpt]

    return (endpt_endpt, intr_intr, g1_intr_g2_endpt, g1_endpt_g2_intr)


def _find_shared_vertex_neighbours(
    analysis: _GraphVertexAnalysis,
    endpts: npt.NDArray[np.integer],
    shared_vert_ids: npt.NDArray[NpIndex],
    context: npt.NDArray[np.bool_],
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """
    Finds adjacency between shared vertices using integer vertex IDs.

    Parameters
    ----------
    analysis : _GraphVertexAnalysis
        Unique-vertex metadata for the graph being inspected.
    endpts : NDArray[int]
        A-by-2 array of inclusive arc ranges in the graph's stored vertex array.
    shared_vert_ids : NDArray[int]
        Unique-vertex IDs in ``analysis`` for every vertex shared by the two
        graphs.
    context : NDArray[bool]
        Mask over the shared vertices selecting those that may participate in
        a shared run.

    Returns
    -------
    prev_alsos : NDArray[bool]
        Whether each shared vertex is immediately preceded in an arc by another
        context vertex.
    after_alsos : NDArray[bool]
        Whether each shared vertex is immediately followed in an arc by another
        context vertex.

    Notes
    -----
    Adjacency is evaluated only between consecutive rows belonging to the same
    arc. Integer lookups avoid repeatedly hashing coordinate tuples.
    """
    prev_alsos = np.zeros(shared_vert_ids.shape[0], dtype=bool)
    after_alsos = np.zeros(shared_vert_ids.shape[0], dtype=bool)
    if (shared_vert_ids.size == 0) or (endpts.shape[0] == 0):
        return prev_alsos, after_alsos

    # Translate graph-local unique IDs to positions in the shared arrays.
    unique_to_shared = np.full(analysis.verts.shape[0], -1, dtype=np.intp)
    unique_to_shared[shared_vert_ids] = np.arange(shared_vert_ids.size)
    vert_shared_ids = np.full(analysis.vert_inv_ids.shape, -1, dtype=np.intp)
    used = analysis.vert_inv_ids >= 0
    vert_shared_ids[used] = unique_to_shared[analysis.vert_inv_ids[used]]

    # A difference array marks consecutive stored rows belonging to one arc.
    segment_counts = np.zeros(analysis.vert_inv_ids.size, dtype=np.int32)
    np.add.at(segment_counts, endpts[:, 0], 1)
    np.add.at(segment_counts, endpts[:, 1], -1)
    same_arc = np.cumsum(segment_counts)[:-1] > 0
    left = vert_shared_ids[:-1]
    right = vert_shared_ids[1:]
    valid = same_arc & (left >= 0) & (right >= 0)
    valid_ids = np.flatnonzero(valid)
    if valid_ids.size:
        valid_ids = valid_ids[context[left[valid_ids]] & context[right[valid_ids]]]
        np.logical_or.at(after_alsos, left[valid_ids], True)
        np.logical_or.at(prev_alsos, right[valid_ids], True)
    return prev_alsos, after_alsos


def _split_arcs_at_vertex_ids(
    orders: npt.NDArray[np.integer],
    ijs: npt.NDArray[np.number],
    endpts: npt.NDArray[NpIndex],
    analysis: _GraphVertexAnalysis,
    split_coord_ids: npt.NDArray[NpIndex],
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.number], npt.NDArray[NpIndex]]:
    """
    Splits all requested interior vertices while rebuilding arrays once.

    Parameters
    ----------
    orders : NDArray[int]
        Order of every input arc.
    ijs : NDArray[int | float]
        V-by-n array of stored vertex coordinates.
    endpts : NDArray[int]
        A-by-2 array of inclusive arc ranges into `ijs`.
    analysis : _GraphVertexAnalysis
        Reusable metadata previously calculated from `ijs` and `endpts`.
    split_coord_ids : NDArray[int]
        Unique-vertex IDs in `analysis` that should become arc endpoints.

    Returns
    -------
    orders : NDArray[int]
        Orders of the rebuilt arcs, with the source order copied to every split portion.
    ijs : NDArray[int | float]
        Compact vertex array containing the rebuilt arcs.
    endpts : NDArray[int]
        Inclusive ranges of the rebuilt arcs in the returned vertex array.

    Notes
    -----
    Every matching interior occurrence is split. The function collects views of all resulting portions and concatenates them once, avoiding repeated growth of the vertex, endpoint, and order arrays.
    """
    if split_coord_ids.size == 0:
        return orders, ijs, endpts

    # Convert unique-coordinate IDs to a mask over stored vertex rows
    split_vertices = np.isin(analysis.vert_inv_ids, split_coord_ids)
    chunks: list[npt.NDArray[np.number]] = []
    new_orders: list[np.generic | int] = []
    lengths: list[int] = []
    for order, (start, end) in zip(orders, endpts):
        # Existing arc endpoints are boundaries already; split only interiors
        interior = np.flatnonzero(split_vertices[start + 1 : end]) + start + 1
        boundaries = np.concatenate(([start], interior, [end]))
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            chunk = ijs[left : right + 1]
            chunks.append(chunk)
            lengths.append(chunk.shape[0])
            new_orders.append(order)

    # Materialise the compact graph once and derive its inclusive arc ranges
    new_ijs = np.concatenate(chunks, axis=0)
    ends = np.cumsum(lengths, dtype=np.intp) - 1
    starts = np.concatenate(([0], ends[:-1] + 1))
    new_endpts = np.column_stack((starts, ends)).astype(endpts.dtype, copy=False)
    return np.asarray(new_orders, dtype=orders.dtype), new_ijs, new_endpts


def solve_graph_overlaps(
    g1_orders: npt.NDArray[np.integer],
    g1_ijs: npt.NDArray[np.number],
    g1_endpts: npt.NDArray[NpIndex],
    g2_orders: npt.NDArray[np.integer],
    g2_ijs: npt.NDArray[np.number],
    g2_endpts: npt.NDArray[NpIndex],
    allows_arcs_overlap: bool = True,
    remove_unused: bool = False,
) -> tuple[
    npt.NDArray[np.integer],
    npt.NDArray[np.number],
    npt.NDArray[NpIndex],
    npt.NDArray[np.integer],
    npt.NDArray[np.number],
    npt.NDArray[NpIndex],
]:
    """
    Splits two graphs at shared vertices to align their arc endpoints.

    Vertices that are endpoints in only one graph are inserted as endpoints in the other graph. Interior overlaps are inserted into both graphs unless they belong to a shared arc and `allows_arcs_overlap` is `True`.

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
        The default option is `True`.
    remove_unused : bool, optional
        Whether to compact both returned vertex arrays so their arc ranges are adjacent.
        Default option is `False`.

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
    g1_analysis = _analyse_graph_vertices(g1_ijs, g1_endpts)
    g2_analysis = _analyse_graph_vertices(g2_ijs, g2_endpts)
    shared = _find_shared_graph_vertices(g1_analysis, g2_analysis)
    g1_ep = shared.g1_is_endpt
    g2_ep = shared.g2_is_endpt
    intr_intr = ~g1_ep & ~g2_ep
    g1_intr_g2_vert = ~g1_ep & g2_ep
    g1_vert_g2_intr = g1_ep & ~g2_ep

    split_both = intr_intr.copy()
    if allows_arcs_overlap and np.any(intr_intr):
        # Coordinates already duplicated across arcs, together with mismatched endpoints that this call will split, bound an existing shared run
        context = intr_intr | (
            (g1_ep & g2_ep & (shared.g1_cnts > 1) & (shared.g2_cnts > 1))
            | (g1_intr_g2_vert & (shared.g2_cnts > 1))
            | (g1_vert_g2_intr & (shared.g1_cnts > 1))
        )
        g1_prev, g1_after = _find_shared_vertex_neighbours(
            g1_analysis, g1_endpts, shared.g1_vert_ids, context
        )
        g2_prev, g2_after = _find_shared_vertex_neighbours(
            g2_analysis, g2_endpts, shared.g2_vert_ids, context
        )
        split_both = intr_intr & ~(g1_prev & g1_after & g2_prev & g2_after)

    g1_split = g1_intr_g2_vert | split_both
    g2_split = g1_vert_g2_intr | split_both
    g1_orders, g1_ijs, g1_endpts = _split_arcs_at_vertex_ids(
        g1_orders,
        g1_ijs,
        g1_endpts,
        g1_analysis,
        shared.g1_vert_ids[g1_split],  # type: ignore
    )
    g2_orders, g2_ijs, g2_endpts = _split_arcs_at_vertex_ids(
        g2_orders,
        g2_ijs,
        g2_endpts,
        g2_analysis,
        shared.g2_vert_ids[g2_split],  # type: ignore
    )
    if remove_unused:
        g1_ijs, g1_endpts = remove_unused_vertices(g1_ijs, g1_endpts)
        g2_ijs, g2_endpts = remove_unused_vertices(g2_ijs, g2_endpts)
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
    orders_list: list[npt.NDArray[np.integer]] | tuple[npt.NDArray[np.integer], ...],
    vertices_list: list[npt.NDArray[np.number]] | tuple[npt.NDArray[np.number], ...],
    endpts_list: list[npt.NDArray[NpIndex]] | tuple[npt.NDArray[NpIndex], ...],
    tol: int | float,
    check_topology: bool,
    backend: str,
) -> tuple[
    list[npt.NDArray[np.integer]] | tuple[npt.NDArray[np.integer], ...],
    list[npt.NDArray[np.number]] | tuple[npt.NDArray[np.number], ...],
    list[npt.NDArray[NpIndex]] | tuple[npt.NDArray[NpIndex], ...],
    list[npt.NDArray[np.bool_]] | tuple[npt.NDArray[np.bool_], ...],
]:
    def is_empty_graph(
        orders: npt.NDArray[np.integer],
        vertices: npt.NDArray[np.number],
        endpts: npt.NDArray[NpIndex],
    ) -> bool:
        return (
            orders.shape == (0,)
            and vertices.shape in ((0, 2), (2, 0))
            and endpts.shape in ((0, 2), (2, 0))
        )

    empty_graphs = [
        is_empty_graph(orders, vertices, endpts)
        for orders, vertices, endpts in zip(orders_list, vertices_list, endpts_list)
    ]
    if len(empty_graphs) == 0 or any(empty_graphs):
        nonempty_ids = [i for i, is_empty in enumerate(empty_graphs) if not is_empty]
        if nonempty_ids:
            nonempty_results = _simplify_multiple_flowgraphs(
                [orders_list[i] for i in nonempty_ids],
                [vertices_list[i] for i in nonempty_ids],
                [endpts_list[i] for i in nonempty_ids],
                tol=tol,
                check_topology=check_topology,
                backend=backend,
            )
        else:
            nonempty_results = ([], [], [], [])

        result_lists: tuple[list[npt.NDArray], ...] = ([], [], [], [])
        nonempty_i = 0
        for i, is_empty in enumerate(empty_graphs):
            if is_empty:
                result_lists[0].append(orders_list[i].copy())
                result_lists[1].append(vertices_list[i].copy())
                result_lists[2].append(endpts_list[i].copy())
                result_lists[3].append(np.empty((0,), dtype=bool))
            else:
                for result_list, nonempty_result in zip(result_lists, nonempty_results):
                    result_list.append(nonempty_result[nonempty_i])
                nonempty_i += 1

        if isinstance(vertices_list, tuple):
            return tuple(tuple(result) for result in result_lists)  # type: ignore
        return result_lists  # type: ignore

    vertices_shps: list[tuple] = []
    endpts_shps: list[tuple] = []

    all_orders_list: list[npt.NDArray[np.integer]] = []
    all_vertices_list: list[npt.NDArray[np.number]] = []
    all_endpts_list: list[npt.NDArray[np.intp]] = []
    all_graph_ids_list: list[npt.NDArray[np.uint8]] = []

    for i, (vertices, endpts, orders) in enumerate(
        zip(vertices_list, endpts_list, orders_list)
    ):
        vertices_shps.append(vertices.shape)
        endpts_shps.append(endpts.shape)

        if vertices.ndim != 2 or endpts.ndim != 2 or orders.ndim != 1:
            raise ValueError(
                f"Graph at index {i} has invalid dimensions (vertices and endpoints "
                "must be 2D arrays, and orders must be a 1D array)."
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
        if orders.shape[0] != endpts.shape[0]:
            raise ValueError(
                f"Order array at index {i} has length {orders.shape[0]}, but the "
                f"endpoint array contains {endpts.shape[0]} arcs."
            )

        all_vertices_list.append(vertices.copy())
        all_endpts_list.append(endpts.copy())
        all_orders_list.append(orders.copy())

    # Insert endpoints at graph overlaps before simplifying any coordinates
    for i in range(len(all_vertices_list) - 1):
        for j in range(i + 1, len(all_vertices_list)):
            (
                all_orders_list[i],
                all_vertices_list[i],
                all_endpts_list[i],
                all_orders_list[j],
                all_vertices_list[j],
                all_endpts_list[j],
            ) = solve_graph_overlaps(
                *(all_orders_list[i], all_vertices_list[i], all_endpts_list[i]),
                *(all_orders_list[j], all_vertices_list[j], all_endpts_list[j]),
                allows_arcs_overlap=True,
            )

    # Concatenate the graphs while retaining the graph membership of each arc
    offset: int = 0
    offset_endpts_list = []
    for i, (vertices, endpts) in enumerate(zip(all_vertices_list, all_endpts_list)):
        offset_endpts_list.append(endpts + offset)
        all_graph_ids_list.append(np.full(endpts.shape[0], i, dtype=np.uint8))
        offset += vertices.shape[0]
    all_orders = np.concatenate(all_orders_list)
    all_vertices = np.concatenate(all_vertices_list, axis=0)
    all_endpts = np.concatenate(offset_endpts_list, axis=0)
    all_graph_ids = np.concatenate(all_graph_ids_list, axis=0)

    # Call the core single flowgraph simplifier
    _, _, _, keeps_concat = _simplify_single_flowgraph(
        *(all_orders, all_vertices, all_endpts),
        tol=tol,
        check_topology=check_topology,
        backend=backend,
        graph_ids=all_graph_ids,
    )

    # Separate the simplified graph back into multiple graphs
    s_orders_list: list[npt.NDArray[np.integer]] = []
    s_vertices_list: list[npt.NDArray[np.number]] = []
    s_endpts_list: list[npt.NDArray[NpIndex]] = []
    keeps_list: list[npt.NDArray[np.bool_]] = []

    offset = 0
    for i in range(len(all_vertices_list)):
        vertex_shp = vertices_shps[i]
        endpts_shp = endpts_shps[i]

        nvertices_i = all_vertices_list[i].shape[0]
        keeps_i = keeps_concat[offset : offset + nvertices_i]

        vertices = all_vertices_list[i]
        simp_v_i = vertices[keeps_i, :]

        vertex_cumsum_i = (np.cumsum(keeps_i) - 1).astype(np.intp)
        local_e_std = all_endpts_list[i]
        simp_e_i = vertex_cumsum_i[local_e_std]

        # Restore original orientation
        if vertex_shp[0] == 2 and vertex_shp[1] != 2:
            simp_v_i = simp_v_i.T
        if endpts_shp[0] == 2 and endpts_shp[1] != 2:
            simp_e_i = simp_e_i.T

        s_orders_list.append(all_orders_list[i])
        s_vertices_list.append(simp_v_i)
        s_endpts_list.append(simp_e_i)  # type: ignore
        keeps_list.append(keeps_i)

        offset += nvertices_i

    if isinstance(vertices_list, tuple):
        return (
            tuple(s_orders_list),
            tuple(s_vertices_list),
            tuple(s_endpts_list),
            tuple(keeps_list),
        )
    else:
        return s_orders_list, s_vertices_list, s_endpts_list, keeps_list


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

    Parameters
    ----------
    vertices : NDArray[int | float]
    endpts : NDArray[int]
    vertex_keeps : NDArray[bool]
    tol : float
    graph_ids : NDArray[int] | None
    max_iters : int
        - max_iters = 0: Validate, but make no repair attempts.
        - max_iters = 1: Make at most one repair attempt.
        - max_iters = N: Make at most N attempts.
    """
    vertex_cumsum = np.cumsum(vertex_keeps) - 1
    vertices_aux = vertices[:, vertex_keeps]
    endpts_aux = vertex_cumsum[endpts]

    intxs = _locate_disallowed_graph_topology(vertices_aux, endpts_aux, graph_ids)

    niters = 0
    while (intxs is not None) and (niters < max_iters):
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
    orders: npt.NDArray[np.integer],
    vertices: npt.NDArray[np.number],
    endpts: npt.NDArray[NpIndex],
    tol: float,
    check_topology: bool,
    backend: str,
    graph_ids: Optional[npt.NDArray[np.integer]] = None,
) -> tuple[
    npt.NDArray[np.integer],
    npt.NDArray[np.number],
    npt.NDArray[NpIndex],
    npt.NDArray[np.bool_],
]:
    """
    Core function to simplify a single flow graph using RDP algorithm.
    """
    if backend != "fortran":
        raise NotImplementedError(
            "Only the FORTRAN backend is implemented at this moment."
        )

    # Standardise inputs to FORTRAN layout (2, N) and (2, A)
    if not (vertices.shape[0] == 2 and vertices.shape[1] != 2):
        vertices = vertices.T
    if not (endpts.shape[0] == 2 and endpts.shape[1] != 2):
        endpts = endpts.T
    if orders.ndim != 1:
        raise ValueError(
            "Orders must be a 1D array, " + f"but has shape {orders.shape}."
        )
    if orders.shape[0] != endpts.shape[1]:
        raise ValueError(
            f"Order array has length {orders.shape[0]}, but the endpoint array "
            f"contains {endpts.shape[1]} arcs."
        )

    # Make a copy of arc_endpts to avoid modifying the input array in-place
    endpts = endpts.copy()

    # Convert 0-based Python indices to 1-based FORTRAN indices
    endpts += 1

    # Call the FORTRAN routine to get the boolean mask of kept vertices
    vertex_keeps: npt.NDArray[np.bool_] = graphs_f.simplify_flowgraph(
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
    simp_endpts = simp_endpts.T.astype(np.intp, order="C")
    return orders.copy(), simp_vertices, simp_endpts, vertex_keeps  # type: ignore


@overload
def simplify_flowgraph(
    arc_orders: npt.NDArray[np.integer],
    vertex_xys: npt.NDArray[np.number],
    arc_endpts: npt.NDArray[NpIndex],
    tol: int | float = 1,
    check_topology: bool = True,
    backend: Literal["fortran", "python"] = "fortran",
    remove_unused: bool = False,
) -> tuple[
    npt.NDArray[np.integer],
    npt.NDArray[np.number],
    npt.NDArray[NpIndex],
    npt.NDArray[np.bool_],
]: ...


@overload
def simplify_flowgraph(
    arc_orders: list[npt.NDArray[np.integer]],
    vertex_xys: list[npt.NDArray[np.number]],
    arc_endpts: list[npt.NDArray[NpIndex]],
    tol: int | float = 1,
    check_topology: bool = True,
    backend: Literal["fortran", "python"] = "fortran",
    remove_unused: bool = False,
) -> tuple[
    list[npt.NDArray[np.integer]],
    list[npt.NDArray[np.number]],
    list[npt.NDArray[NpIndex]],
    list[npt.NDArray[np.bool_]],
]: ...


@overload
def simplify_flowgraph(
    arc_orders: tuple[npt.NDArray[np.integer], ...],
    vertex_xys: tuple[npt.NDArray[np.number], ...],
    arc_endpts: tuple[npt.NDArray[NpIndex], ...],
    tol: int | float = 1,
    check_topology: bool = True,
    backend: Literal["fortran", "python"] = "fortran",
    remove_unused: bool = False,
) -> tuple[
    tuple[npt.NDArray[np.integer], ...],
    tuple[npt.NDArray[np.number], ...],
    tuple[npt.NDArray[NpIndex], ...],
    tuple[npt.NDArray[np.bool_], ...],
]: ...


def simplify_flowgraph(
    arc_orders: (
        npt.NDArray[np.integer]
        | list[npt.NDArray[np.integer]]
        | tuple[npt.NDArray[np.integer], ...]
    ),
    vertex_xys: (
        npt.NDArray[np.number]
        | list[npt.NDArray[np.number]]
        | tuple[npt.NDArray[np.number], ...]
    ),
    arc_endpts: (
        npt.NDArray[NpIndex]
        | list[npt.NDArray[NpIndex]]
        | tuple[npt.NDArray[NpIndex], ...]
    ),
    tol: int | float = 1,
    check_topology: bool = True,
    backend: Literal["fortran", "python"] = "fortran",
    remove_unused: bool = False,
):
    """
    Simplify a flow graph using the Ramer-Douglas-Peucker (RDP) algorithm with a fixed tolerance threshold.

    When multiple graphs are supplied, their overlaps are first split into compatible arcs using `solve_graph_overlaps`. Identical arcs belonging to different graphs are ignored during topology validation, including when their vertex directions are reversed.

    Parameters
    ----------
    arc_orders : NDArray[int] or Iterable[NDArray[int]]
        A-by-(1) array containing the order of each arc, or an iterable of such arrays.
    vertex_xys : NDArray[number] or Iterable[NDArray[number]]
        V-by-2 (or 2-by-V) array of coordinates representing the vertices in the flow graph, or an iterable of such arrays.
    arc_endpts : NDArray[int] or Iterable[NDArray[int]]
        A-by-2 (or 2-by-A) array of indices indicating where each arc starts and ends in `vertex_xys`, or an iterable of such arrays.
    tol : int | float, optional
        Tolerance threshold for simplification.
        Vertices with perpendicular distance to the line segment less than or equal to `tol` will be simplified/removed.
        Default tolerance is 1.
    check_topology : bool, optional
        Whether to check for invalid topography in the simplified graph.
        Default option is `True`.
    backend : {'fortran', 'python'}, optional
        Backend to use for computation.
        Default backend and the only one currently available is `'fortran'`.
    remove_unused : bool, optional
        Whether to compact each returned vertex array so its arc ranges are adjacent.
        Default option is `False`.

    Returns
    -------
    simp_arc_orders : NDArray[int] or list/tuple of NDArray[int]
        Order of every simplified graph arc, including arcs introduced while aligning graph overlaps.
    simp_vertex_xys : NDArray[number] or list/tuple of NDArray[number]
        V'-by-2 array of coordinates representing the simplified vertices, or a list/tuple of such arrays.
    simp_arc_endpts : NDArray[int32] or list/tuple of NDArray[int32]
        A-by-2 array of indices indicating the start and end of each simplified arc, or a list/tuple of such arrays.
    keeps : NDArray[bool] or list/tuple of NDArray[bool]
        V-by-1 mask indicating which of the input vertices are retained in the simplified graph, or a list/tuple of such masks.
        For multiple overlapping graphs, the masks refer to the intermediate vertex arrays produced by `solve_graph_overlaps`, which may contain additional vertices.

    Raises
    ------
    TypeError
        If a single graph is supplied and any of `arc_orders`, `vertex_xys`, or `arc_endpts` is not a NumPy array.
    ValueError
         1. If single-graph and multi-graph argument forms are mixed.
         2. If the multi-graph argument collections have different lengths.
         3. If an order array is not one-dimensional or does not contain one value per arc.
         4. If a vertex or endpoint array has an invalid shape.
    InvalidOriginalGraphTopology
        If the final result is invalid and the normalised input graph already contains disallowed topology violations.
    UnresolvedSimplificationTopology
        If the normalised input is valid but the final simplified graph contains disallowed topology violations.
    NotImplementedError
        If tries to call the not-yet-implemented Python backend.
    """

    is_multi = (
        isinstance(vertex_xys, (list, tuple))
        or isinstance(arc_endpts, (list, tuple))
        or isinstance(arc_orders, (list, tuple))
    )
    if is_multi:
        if (
            (not isinstance(vertex_xys, (list, tuple)))
            or (not isinstance(arc_endpts, (list, tuple)))
            or (not isinstance(arc_orders, (list, tuple)))
        ):
            raise ValueError(
                "Arguments 'vertex_xys', 'arc_endpts', and 'arc_orders' must all "
                "be iterables (or none of them)."
            )
        if not (len(vertex_xys) == len(arc_endpts) == len(arc_orders)):
            raise ValueError(
                "Arguments 'vertex_xys', 'arc_endpts', and 'arc_orders' must have "
                f"the same length, but got {len(vertex_xys)}, {len(arc_endpts)}, "
                f"and {len(arc_orders)}, respectively."
            )
        result = _simplify_multiple_flowgraphs(
            *(arc_orders, vertex_xys, arc_endpts),
            tol=tol,
            check_topology=check_topology,
            backend=backend,
        )
        if not remove_unused:
            return result
        simp_orders, simp_vertices, simp_endpts, keeps = result
        compacted = [
            remove_unused_vertices(vertices, endpts)
            for vertices, endpts in zip(simp_vertices, simp_endpts)
        ]
        compact_vertices = type(simp_vertices)(item[0] for item in compacted)
        compact_endpts = type(simp_endpts)(item[1] for item in compacted)
        return simp_orders, compact_vertices, compact_endpts, keeps

    if not (
        isinstance(arc_orders, np.ndarray)
        and isinstance(vertex_xys, np.ndarray)
        and isinstance(arc_endpts, np.ndarray)
    ):
        raise TypeError(
            "Arguments 'vertex_xys', 'arc_endpts', and 'arc_orders' must be "
            "NumPy arrays, but got "
            f"{type(vertex_xys)}, {type(arc_endpts)}, and {type(arc_orders)}, "
            "respectively."
        )
    result = _simplify_single_flowgraph(
        *(arc_orders, vertex_xys, arc_endpts),
        tol=tol,
        check_topology=check_topology,
        backend=backend,
    )
    if not remove_unused:
        return result
    simp_orders, simp_vertices, simp_endpts, keeps = result
    simp_vertices, simp_endpts = remove_unused_vertices(simp_vertices, simp_endpts)
    return simp_orders, simp_vertices, simp_endpts, keeps


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
    capacity = max(
        vertices_f.shape[1] // 100, 3
    )  # Arbitrary capacity that seems to work

    intxs, nintxs, err_code = graphs_f.scan_invalid_graph_topology(
        vertices_f, endpts_f, capacity
    )
    raise_fortran_error(
        "scan_invalid_graph_topology",
        err_code,
    )

    if nintxs == 0:
        return None

    if nintxs > capacity:
        expected_nintxs = nintxs
        intxs, nintxs, err_code = graphs_f.scan_invalid_graph_topology(
            vertices_f, endpts_f, expected_nintxs
        )
        raise_fortran_error(
            "scan_invalid_graph_topology",
            err_code,
        )
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
