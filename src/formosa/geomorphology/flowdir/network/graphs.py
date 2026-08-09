# Last modified
#   2026-07-09, En-Chi Lee (williameclee@gmail.com)
#     - Added vertex mask to output of function
#       `simplify_flowgraph`
#   2026-07-13, En-Chi Lee (williameclee@gmail.com)
#     - Added default topology check to `simplify_flowgraph`
#   2026-07-14, En-Chi Lee (williameclee@gmail.com)
#     - Added simultaneous multi-graph checks to
#       `simplify_flowgraph`
#   2026-07-28, En-Chi Lee (williameclee@gmail.com)
#     - Implemented `solve_graph_overlaps` and relevant helper
#       functions
#     - Integrated overlap resolution into simultaneous multi-
#       graph simplification
#   2026-07-29, En-Chi Lee (williameclee@gmail.com)
#     - Added validation to simplified graph before return
#   2026-07-31, En-Chi Lee (williameclee@gmail.com)
#     - Preserved arc orders when `simplify_flowgraph` splits
#       overlapping graphs
#   2026-08-03, En-Chi Lee (williameclee@gmail.com)
#     - Made `solve_graph_overlaps` stable and recognised already
#       valid shared arcs
#   2026-08-04, En-Chi Lee (williameclee@gmail.com)
#     - Made `simplify_flowgraph` able to handle empty graphs
#     - Accelerated graph validation and simplification


from dataclasses import dataclass
import numpy as np

from formosa.geomorphology.flowdir.network.validation import (
    InvalidOriginalGraphTopology,
    UnresolvedSimplificationTopology,
    locate_invalid_graph_topology,
)
from formosa.geomorphology.flowdir.network.editing import remove_unused_vertices
from formosa.geomorphology.flowdir_f import flowdir_graphs as graphs_f

import numpy.typing as npt
from typing import Literal, Optional, TypeVar, overload

NpIndex = TypeVar("NpIndex", np.int32, np.int64, np.intp)


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
