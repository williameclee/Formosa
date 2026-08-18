"""
Simplifies flow-graph arcs while preserving valid topology.

Last modified: 2026-08-18, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np

from formosa.utils import Backend
from formosa.geomorphology.drainage.network.overlaps import (
    _resolve_topology_intersections,
    solve_graph_overlaps,
)
from formosa.geomorphology.drainage.network.validation import (
    InvalidOriginalGraphTopology,
    UnresolvedSimplificationTopology,
    _locate_disallowed_graph_topology,
)
from formosa.geomorphology.drainage.network.editing import remove_unused_vertices
from formosa.geomorphology._native import network_simplification as simp_f

from numpy.typing import NDArray
from typing import Optional, overload
from formosa.utils import NpInt, NpIndex, NpCoords


def _convert_index_array_to_F_fmt(vertices: NDArray) -> NDArray:
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
    orders_list: list[NDArray[NpInt]] | tuple[NDArray[NpInt], ...],
    vtxs_list: list[NDArray[NpCoords]] | tuple[NDArray[NpCoords], ...],
    endpts_list: list[NDArray[NpIndex]] | tuple[NDArray[NpIndex], ...],
    tol: int | float,
    check_topology: bool,
    backend: Backend,
) -> tuple[
    list[NDArray[NpInt]] | tuple[NDArray[NpInt], ...],
    list[NDArray[NpCoords]] | tuple[NDArray[NpCoords], ...],
    list[NDArray[NpIndex]] | tuple[NDArray[NpIndex], ...],
    list[NDArray[np.bool_]] | tuple[NDArray[np.bool_], ...],
]:
    def is_empty_graph(
        orders: NDArray[np.integer],
        vtxs: NDArray[NpCoords],
        endpts: NDArray[NpIndex],
    ) -> bool:
        return (
            orders.shape == (0,)
            and vtxs.shape in ((0, 2), (2, 0))
            and endpts.shape in ((0, 2), (2, 0))
        )

    empty_graphs = [
        is_empty_graph(orders, vtxs, endpts)
        for orders, vtxs, endpts in zip(orders_list, vtxs_list, endpts_list)
    ]
    if len(empty_graphs) == 0 or any(empty_graphs):
        nonempty_ids = [i for i, is_empty in enumerate(empty_graphs) if not is_empty]
        if nonempty_ids:
            nonempty_results = _simplify_multiple_flowgraphs(
                [orders_list[i] for i in nonempty_ids],
                [vtxs_list[i] for i in nonempty_ids],
                [endpts_list[i] for i in nonempty_ids],
                tol=tol,
                check_topology=check_topology,
                backend=backend,
            )
        else:
            nonempty_results = ([], [], [], [])

        result_lists: tuple[list[NDArray], ...] = ([], [], [], [])
        nonempty_i = 0
        for i, is_empty in enumerate(empty_graphs):
            if is_empty:
                result_lists[0].append(orders_list[i].copy())
                result_lists[1].append(vtxs_list[i].copy())
                result_lists[2].append(endpts_list[i].copy())
                result_lists[3].append(np.empty((0,), dtype=bool))
            else:
                for result_list, nonempty_result in zip(result_lists, nonempty_results):
                    result_list.append(nonempty_result[nonempty_i])
                nonempty_i += 1

        if isinstance(vtxs_list, tuple):
            return tuple(tuple(result) for result in result_lists)  # type: ignore
        return result_lists  # type: ignore

    vtx_shps: list[tuple] = []
    endpts_shps: list[tuple] = []

    all_orders_list: list[NDArray[NpInt]] = []
    all_vtxs_list: list[NDArray[NpCoords]] = []
    all_endpts_list: list[NDArray[NpIndex]] = []
    all_graph_ids_list: list[NDArray[np.uint8]] = []

    for i, (vtxs, endpts, orders) in enumerate(
        zip(vtxs_list, endpts_list, orders_list)
    ):
        vtx_shps.append(vtxs.shape)
        endpts_shps.append(endpts.shape)

        if vtxs.ndim != 2 or endpts.ndim != 2 or orders.ndim != 1:
            raise ValueError(
                f"Graph at index {i} has invalid dimensions (vertices and endpoints must be 2D arrays, and orders must be a 1D array)."
            )
        # Standardise vertices and endpoints arrays
        try:
            vtxs = _convert_index_array_to_F_fmt(vtxs)
        except ValueError:
            raise ValueError(
                f"Vertex array at index {i} shape {vtxs.shape} is not V-by-2 or 2-by-V."
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

        all_vtxs_list.append(vtxs.copy())
        all_endpts_list.append(endpts.copy())
        all_orders_list.append(orders.copy())

    # Insert endpoints at graph overlaps before simplifying any coordinates
    for i in range(len(all_vtxs_list) - 1):
        for j in range(i + 1, len(all_vtxs_list)):
            (
                all_orders_list[i],
                all_vtxs_list[i],
                all_endpts_list[i],
                all_orders_list[j],
                all_vtxs_list[j],
                all_endpts_list[j],
            ) = solve_graph_overlaps(
                *(all_orders_list[i], all_vtxs_list[i], all_endpts_list[i]),
                *(all_orders_list[j], all_vtxs_list[j], all_endpts_list[j]),
                allows_arcs_overlap=True,
            )

    # Concatenate the graphs while retaining the graph membership of each arc
    offset: int = 0
    offset_endpts_list = []
    for i, (vtxs, endpts) in enumerate(zip(all_vtxs_list, all_endpts_list)):
        offset_endpts_list.append(endpts + offset)
        all_graph_ids_list.append(np.full(endpts.shape[0], i, dtype=np.uint8))
        offset += vtxs.shape[0]
    all_orders = np.concatenate(all_orders_list)
    all_vtxs = np.concatenate(all_vtxs_list, axis=0)
    all_endpts = np.concatenate(offset_endpts_list, axis=0)
    all_graph_ids = np.concatenate(all_graph_ids_list, axis=0)

    # Call the core single flowgraph simplifier
    _, _, _, keeps_concat = _simplify_single_flowgraph(
        *(all_orders, all_vtxs, all_endpts),
        tol=tol,
        check_topology=check_topology,
        backend=backend,
        graph_ids=all_graph_ids,
    )

    # Separate the simplified graph back into multiple graphs
    s_orders_list: list[NDArray[NpInt]] = []
    s_vtxs_list: list[NDArray[NpCoords]] = []
    s_endpts_list: list[NDArray[NpIndex]] = []
    keeps_list: list[NDArray[np.bool_]] = []

    offset = 0
    for i in range(len(all_vtxs_list)):
        vertex_shp = vtx_shps[i]
        endpts_shp = endpts_shps[i]

        nvtxs_i = all_vtxs_list[i].shape[0]
        keeps_i = keeps_concat[offset : offset + nvtxs_i]

        vtxs = all_vtxs_list[i]
        simp_v_i = vtxs[keeps_i, :]

        vtx_cumsum_i = (np.cumsum(keeps_i) - 1).astype(np.intp)
        local_e_std = all_endpts_list[i]
        simp_e_i = vtx_cumsum_i[local_e_std]

        # Restore original orientation
        if vertex_shp[0] == 2 and vertex_shp[1] != 2:
            simp_v_i = simp_v_i.T
        if endpts_shp[0] == 2 and endpts_shp[1] != 2:
            simp_e_i = simp_e_i.T

        s_orders_list.append(all_orders_list[i])
        s_vtxs_list.append(simp_v_i)
        s_endpts_list.append(simp_e_i)  # type: ignore
        keeps_list.append(keeps_i)

        offset += nvtxs_i

    if isinstance(vtxs_list, tuple):
        return (
            tuple(s_orders_list),
            tuple(s_vtxs_list),
            tuple(s_endpts_list),
            tuple(keeps_list),
        )
    else:
        return s_orders_list, s_vtxs_list, s_endpts_list, keeps_list


def _simplify_single_flowgraph(
    orders: NDArray[np.integer],
    vtxs: NDArray[NpCoords],
    endpts: NDArray[NpIndex],
    tol: float,
    check_topology: bool,
    backend: Backend,
    graph_ids: Optional[NDArray[np.integer]] = None,
) -> tuple[
    NDArray[np.integer],
    NDArray[NpCoords],
    NDArray[NpIndex],
    NDArray[np.bool_],
]:
    """
    Core function to simplify a single flow graph using RDP algorithm.
    """
    if backend != "fortran":
        raise NotImplementedError(
            "Only the Fortran backend is implemented at this moment."
        )

    # Standardise inputs to Fortran layout (2, N) and (2, A)
    if not (vtxs.shape[0] == 2 and vtxs.shape[1] != 2):
        vtxs = vtxs.T
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

    # Convert 0-based Python indices to 1-based Fortran indices
    endpts += 1

    # Call the Fortran routine to get the boolean mask of kept vertices
    vtx_keeps: NDArray[np.bool_] = simp_f.simplify_flowgraph(
        vtxs.astype(np.float32, order="F"),
        endpts.astype(np.int32, order="F"),
        tol,
    ).astype(bool)

    # Revert back to 0-based Python indexing
    endpts -= 1

    if check_topology:
        vtx_keeps = _resolve_topology_intersections(
            vtxs, endpts, vtx_keeps, tol, graph_ids=graph_ids
        )

    # Squeeze the vertices and map the arc endpoints to the new indices
    vtx_cumsum = np.cumsum(vtx_keeps) - 1
    simp_vtxs = vtxs[:, vtx_keeps]
    simp_endpts = vtx_cumsum[endpts]

    if check_topology:
        final_intxs = _locate_disallowed_graph_topology(
            simp_vtxs, simp_endpts, graph_ids
        )
        if final_intxs is not None:
            input_intxs = _locate_disallowed_graph_topology(vtxs, endpts, graph_ids)
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
    simp_vtxs = simp_vtxs.T.astype(vtxs.dtype, order="C")
    simp_endpts = simp_endpts.T.astype(np.intp, order="C")
    return orders.copy(), simp_vtxs, simp_endpts, vtx_keeps  # type: ignore


@overload
def simplify_flowgraph(
    arc_orders: NDArray[NpInt],
    vtx_xys: NDArray[NpCoords],
    arc_endpts: NDArray[NpIndex],
    tol: int | float = 1,
    check_topology: bool = True,
    remove_unused: bool = False,
    backend: Backend = "fortran",
) -> tuple[
    NDArray[NpInt],
    NDArray[NpCoords],
    NDArray[NpIndex],
    NDArray[np.bool_],
]: ...


@overload
def simplify_flowgraph(
    arc_orders: list[NDArray[NpInt]],
    vtx_xys: list[NDArray[NpCoords]],
    arc_endpts: list[NDArray[NpIndex]],
    tol: int | float = 1,
    check_topology: bool = True,
    remove_unused: bool = False,
    backend: Backend = "fortran",
) -> tuple[
    list[NDArray[NpInt]],
    list[NDArray[NpCoords]],
    list[NDArray[NpIndex]],
    list[NDArray[np.bool_]],
]: ...


@overload
def simplify_flowgraph(
    arc_orders: tuple[NDArray[NpInt], ...],
    vtx_xys: tuple[NDArray[NpCoords], ...],
    arc_endpts: tuple[NDArray[NpIndex], ...],
    tol: int | float = 1,
    check_topology: bool = True,
    remove_unused: bool = False,
    backend: Backend = "fortran",
) -> tuple[
    tuple[NDArray[NpInt], ...],
    tuple[NDArray[NpCoords], ...],
    tuple[NDArray[NpIndex], ...],
    tuple[NDArray[np.bool_], ...],
]: ...


def simplify_flowgraph(
    arc_orders: NDArray[NpInt] | list[NDArray[NpInt]] | tuple[NDArray[NpInt], ...],
    vtx_xys: (
        NDArray[NpCoords] | list[NDArray[NpCoords]] | tuple[NDArray[NpCoords], ...]
    ),
    arc_endpts: (
        NDArray[NpIndex] | list[NDArray[NpIndex]] | tuple[NDArray[NpIndex], ...]
    ),
    tol: int | float = 1,
    check_topology: bool = True,
    remove_unused: bool = False,
    backend: Backend = "fortran",
):
    """
    Simplifies a flow graph using the Ramer-Douglas-Peucker (RDP)
    algorithm with a fixed tolerance threshold.

    When multiple graphs are supplied, their overlaps are first
    split into compatible arcs using `solve_graph_overlaps`.
    Identical arcs belonging to different graphs are ignored during
    topology validation, including when their vertex directions are
    reversed.

    Parameters
    ----------
    arc_orders : NDArray[int] or Iterable[NDArray[int]]
        (A,) array containing the order of each arc, or an iterable
        of such arrays.
    vtx_xys : NDArray[number] or Iterable[NDArray[number]]
        (V,2) array of coordinates representing the vertices in the
        flow graph, or an iterable of such arrays.
    arc_endpts : NDArray[int] or Iterable[NDArray[int]]
        (A,2) array of indices indicating where each arc starts and
        ends in `vtx_xys`, or an iterable of such arrays.
    tol : int | float, optional
        Tolerance threshold for simplification.
        Vertices with perpendicular distance to the line segment
        less than or equal to `tol` will be simplified/removed.
        Default tolerance is 1.
    check_topology : bool, optional
        Whether to check for invalid topography in the simplified
        graph.
        Default option is `True`.
    remove_unused : bool, optional
        Whether to compact each returned vertex array so its arc
        ranges are adjacent.
        Default option is `False`.
    backend : {'fortran', 'python'}, optional
        Backend to use for computation.
        `'fortran'` uses the Fortran extension for performance,
        while `'python'` uses a pure Python implementation.
        Default backend is `'fortran'`.

    Returns
    -------
    simp_arc_orders : NDArray[int] or list/tuple of NDArray[int]
        Order of every simplified graph arc, including arcs
        introduced while aligning graph overlaps.
    simp_vtx_xys : NDArray[number] or list/tuple of NDArray[number]
        (V',2) array of coordinates representing the simplified
        vertices, or a list/tuple of such arrays.
    simp_arc_endpts : NDArray[int32] or list/tuple of NDArray[int32]
        (A,2) array of indices indicating the start and end of each
        simplified arc, or a list/tuple of such arrays.
    keeps : NDArray[bool] or list/tuple of NDArray[bool]
        (V,) mask indicating which of the input vertices are
        retained in the simplified graph, or a list/tuple of such
        masks.
        For multiple overlapping graphs, the masks refer to the
        intermediate vertex arrays produced by
        :func:`solve_graph_overlaps`, which may contain additional
        vertices.

    Raises
    ------
    TypeError
        If a single graph is supplied and any of param`arc_orders`,
        `vtx_xys`, or `arc_endpts` is not a NumPy array.
    ValueError
        1. If single-graph and multi-graph argument forms are mixed.
        2. If the multi-graph argument collections have different
            lengths.
        3. If an order array is not one-dimensional or does not
            contain one value per arc.
        4. If a vertex or endpoint array has an invalid shape.
    InvalidOriginalGraphTopology
        If the final result is invalid and the normalised input
        graph already contains disallowed topology violations.
    UnresolvedSimplificationTopology
        If the normalised input is valid but the final simplified
        graph contains disallowed topology violations.
    NotImplementedError
        If tries to call the not-yet-implemented Python backend.
    """

    is_multi = (
        isinstance(vtx_xys, (list, tuple))
        or isinstance(arc_endpts, (list, tuple))
        or isinstance(arc_orders, (list, tuple))
    )
    if is_multi:
        if (
            (not isinstance(vtx_xys, (list, tuple)))
            or (not isinstance(arc_endpts, (list, tuple)))
            or (not isinstance(arc_orders, (list, tuple)))
        ):
            raise ValueError(
                "Arguments 'vtx_xys', 'arc_endpts', and 'arc_orders' must all be iterables (or none of them)."
            )
        if not (len(vtx_xys) == len(arc_endpts) == len(arc_orders)):
            raise ValueError(
                "Arguments 'vtx_xys', 'arc_endpts', and 'arc_orders' must have "
                f"the same length, but got {len(vtx_xys)}, {len(arc_endpts)}, "
                f"and {len(arc_orders)}, respectively."
            )
        result = _simplify_multiple_flowgraphs(
            *(arc_orders, vtx_xys, arc_endpts),
            tol=tol,
            check_topology=check_topology,
            backend=backend,
        )
        if not remove_unused:
            return result
        simp_orders, simp_vtxs, simp_endpts, keeps = result
        compacted = [
            remove_unused_vertices(vertices, endpts)
            for vertices, endpts in zip(simp_vtxs, simp_endpts)
        ]
        compact_vtxs = type(simp_vtxs)(item[0] for item in compacted)
        compact_endpts = type(simp_endpts)(item[1] for item in compacted)
        return simp_orders, compact_vtxs, compact_endpts, keeps

    if not (
        isinstance(arc_orders, np.ndarray)
        and isinstance(vtx_xys, np.ndarray)
        and isinstance(arc_endpts, np.ndarray)
    ):
        raise TypeError(
            "Arguments 'vtx_xys', 'arc_endpts', and 'arc_orders' must be NumPy arrays, "
            f"but got {type(vtx_xys)}, {type(arc_endpts)}, and {type(arc_orders)}, respectively."
        )
    result = _simplify_single_flowgraph(
        *(arc_orders, vtx_xys, arc_endpts),
        tol=tol,
        check_topology=check_topology,
        backend=backend,
    )
    if not remove_unused:
        return result
    simp_orders, simp_vtxs, simp_endpts, keeps = result
    simp_vtxs, simp_endpts = remove_unused_vertices(simp_vtxs, simp_endpts)
    return simp_orders, simp_vtxs, simp_endpts, keeps
