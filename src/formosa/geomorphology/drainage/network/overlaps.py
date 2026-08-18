"""
Detects and resolves overlapping or crossing flow-graph topology.

Last modified: 2026-08-18, En-Chi Lee (williameclee@gmail.com)
"""

from dataclasses import dataclass
import numpy as np

from formosa.geomorphology.drainage.network.validation import (
    _locate_disallowed_graph_topology,
)
from formosa.geomorphology.drainage.network.editing import remove_unused_vertices
from formosa.geomorphology._native import network_simplification as simp_f

from typing import Optional, Generic
from numpy.typing import NDArray
from formosa.utils.typing import NpInt, NpIndex, NpCoords


@dataclass
class _GraphVertexAnalysis(Generic[NpCoords]):
    """
    Reusable vertex metadata for one flow graph.

    Attributes
    ----------
    vtxs : NDArray[int | float]
        Unique vertex coordinates referenced by at least one graph
        arc.
    is_endpt : NDArray[bool]
        Whether any occurrence of each vertex in `vtxs` is an arc
        endpoint.
    vtx_cnts : NDArray[int]
        Number of occurrences of each vertex within the graph's arc
        ranges.
    vtx_inv_ids : NDArray[int]
        Maps each row of the original vertex array to its row in
        `vtxs`.
        Stored vertices that are not referenced by an arc are
        assigned `-1`.
    """

    vtxs: NDArray[NpCoords]
    is_endpt: NDArray[np.bool_]
    vtx_cnts: NDArray[np.intp]
    vtx_inv_ids: NDArray[np.intp]


@dataclass
class _SharedVertexAnalysis(Generic[NpCoords]):
    """
    Vertex metadata shared by a pair of flow graphs.

    Attributes
    ----------
    vtxs : NDArray[int | float]
        Unique vertex coordinates referenced by both graphs.
    g1_vtx_ids : NDArray[int]
        Maps each row in `vtxs` to its unique-vertex ID in the first
        graph.
    g2_vtx_ids : NDArray[int]
        Maps each row in `vtxs` to its unique-vertex ID in the
        second graph.
    g1_is_endpt : NDArray[bool]
        Whether each shared vertex is an endpoint in the first
        graph.
    g2_is_endpt : NDArray[bool]
        Whether each shared vertex is an endpoint in the second
        graph.
    g1_cnts : NDArray[int]
        Number of occurrences of each shared vertex in the first
        graph.
    g2_cnts : NDArray[int]
        Number of occurrences of each shared vertex in the second
        graph.
    """

    vtxs: NDArray[NpCoords]
    g1_vtx_ids: NDArray[np.intp]
    g2_vtx_ids: NDArray[np.intp]
    g1_is_endpt: NDArray[np.bool_]
    g2_is_endpt: NDArray[np.bool_]
    g1_cnts: NDArray[np.intp]
    g2_cnts: NDArray[np.intp]


def _analyse_graph_vertices(
    vtxs: NDArray[NpCoords], endpts: NDArray[NpIndex]
) -> _GraphVertexAnalysis:
    """
    Builds reusable vertex IDs, endpoint roles, and occurrence
    counts.

    Only rows referenced by an inclusive range in `endpts`
    contribute to the unique vertices, roles, and counts. The
    inverse-ID array retains the shape of the original stored vertex
    array so later operations can work directly with arc endpoint
    indices.

    Parameters
    ----------
    vtxs : NDArray[int | float]
        (V,n) array of stored vertex coordinates.
    endpts : NDArray[int]
        (A,2) array of inclusive arc ranges into `vtxs`.

    Returns
    -------
    _GraphVertexAnalysis
        Unique referenced vertices and their reusable graph
        metadata.
    """
    endpts = np.asarray(endpts)
    nvtxs = vtxs.shape[0]
    if endpts.shape[0] == 0:
        return _GraphVertexAnalysis(
            vtxs=np.empty((0, vtxs.shape[1]), dtype=vtxs.dtype),
            is_endpt=np.empty((0,), dtype=bool),
            vtx_cnts=np.empty((0,), dtype=np.intp),
            vtx_inv_ids=np.full(vtxs.shape[0], -1, dtype=np.intp),
        )

    # Expand the inclusive arc ranges, excluding stored rows unused by any arc
    used_ids: NDArray[np.integer] = np.concatenate(
        [np.arange(start, end + 1) for start, end in endpts]
    )
    endpt_ids = np.concatenate((endpts[:, 0], endpts[:, 1]))
    used_is_endpt = np.isin(used_ids, endpt_ids)

    # `vert_inv` maps each used occurrence to its deduplicated vertex
    vtxs, vtx_inv, vtx_cnts = np.unique(
        vtxs[used_ids], axis=0, return_inverse=True, return_counts=True
    )
    # A repeated coordinate is an endpoint if any of its occurrences is one
    is_endpt = np.zeros(vtxs.shape[0], dtype=bool)
    np.logical_or.at(is_endpt, vtx_inv, used_is_endpt)

    # Restore a lookup indexed like the original stored vertex array
    vtx_inv_ids = np.full(nvtxs, -1, dtype=np.intp)
    vtx_inv_ids[used_ids] = vtx_inv
    return _GraphVertexAnalysis(
        vtxs=vtxs,
        is_endpt=is_endpt,
        vtx_cnts=vtx_cnts.astype(np.intp, copy=False),
        vtx_inv_ids=vtx_inv_ids,
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
        Shared vertices, their unique-vertex IDs in both graphs,
        endpoint roles, and occurrence counts.

    Notes
    -----
    Vertex coordinate rows are viewed as fixed-width byte values so
    NumPy can perform a 1D set intersection without Python tuple
    conversion.
    """
    dtype = np.result_type(g1.vtxs.dtype, g2.vtxs.dtype)
    g1_verts = np.ascontiguousarray(g1.vtxs, dtype=dtype)
    g2_verts = np.ascontiguousarray(g2.vtxs, dtype=dtype)
    # Encode each vertex row as one fixed-width value for `intersect1d`.
    row_dtype = np.dtype((np.void, dtype.itemsize * g1_verts.shape[1]))
    g1_keys = g1_verts.view(row_dtype).ravel()
    g2_keys = g2_verts.view(row_dtype).ravel()
    _, g1_ids, g2_ids = np.intersect1d(g1_keys, g2_keys, return_indices=True)
    return _SharedVertexAnalysis(
        vtxs=g1_verts[g1_ids],
        g1_vtx_ids=g1_ids.astype(np.intp, copy=False),
        g2_vtx_ids=g2_ids.astype(np.intp, copy=False),
        g1_is_endpt=g1.is_endpt[g1_ids],
        g2_is_endpt=g2.is_endpt[g2_ids],
        g1_cnts=g1.vtx_cnts[g1_ids],
        g2_cnts=g2.vtx_cnts[g2_ids],
    )


def find_graph_overlaps(
    g1_vtxs: NDArray[NpCoords],
    g1_endpts: NDArray[NpIndex],
    g2_vtxs: NDArray[NpCoords],
    g2_endpts: NDArray[NpIndex],
) -> tuple[NDArray[NpCoords], NDArray[NpCoords], NDArray[NpCoords], NDArray[NpCoords]]:
    """
    Finds and classifies the shared vertices of two graphs.

    Only vertices referenced by the inclusive arc ranges in
    `g1_endpts` and `g2_endpts` are considered. Repeated coordinates
    within a graph are treated as endpoints when at least one
    occurrence is an endpoint.

    Parameters
    ----------
    g1_vtxs : NDArray[int | float]
        (V1,n) array containing the coordinates of all vertices
        stored for the first graph.
    g1_endpts : NDArray[int]
        (A1,2) array containing the inclusive starting and ending
        vertex indices of each arc in the first graph.
    g2_vtxs : NDArray[int | float]
        (V2,n) array containing the coordinates of all vertices
        stored for the second graph.
    g2_endpts : NDArray[int]
        (A2,2) array containing the inclusive starting and ending
        vertex indices of each arc in the second graph.

    Returns
    -------
    endpt_endpt_ijs : NDArray[int | float]
        Coordinates that are endpoints in both graphs.
    intr_intr_ijs : NDArray[int | float]
        Coordinates that are interior vertices in both graphs.
    g1_intr_g2_endpt_ijs : NDArray[int | float]
        Coordinates that are interior vertices in the first graph
        and endpoints in the second graph.
    g1_endpt_g2_intr_ijs : NDArray[int | float]
        Coordinates that are endpoints in the first graph and
        interior vertices in the second graph.
    """
    shared = _find_shared_graph_vertices(
        _analyse_graph_vertices(g1_vtxs, g1_endpts),
        _analyse_graph_vertices(g2_vtxs, g2_endpts),
    )
    overlaps = shared.vtxs.astype(g1_vtxs.dtype, copy=False)

    # Partition the overlaps by their roles in the two graphs
    endpt_endpt = overlaps[shared.g1_is_endpt & shared.g2_is_endpt]
    intr_intr = overlaps[~shared.g1_is_endpt & ~shared.g2_is_endpt]
    g1_intr_g2_endpt = overlaps[~shared.g1_is_endpt & shared.g2_is_endpt]
    g1_endpt_g2_intr = overlaps[shared.g1_is_endpt & ~shared.g2_is_endpt]

    return (endpt_endpt, intr_intr, g1_intr_g2_endpt, g1_endpt_g2_intr)


def _find_shared_vertex_neighbours(
    analysis: _GraphVertexAnalysis,
    endpts: NDArray[np.integer],
    shared_vtx_ids: NDArray[NpIndex],
    context: NDArray[np.bool_],
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """
    Finds adjacency between shared vertices using integer vertex IDs.

    Parameters
    ----------
    analysis : _GraphVertexAnalysis
        Unique-vertex metadata for the graph being inspected.
    endpts : NDArray[int]
        (A,2) array of inclusive arc ranges in the graph's stored
        vertex array.
    shared_vtx_ids : NDArray[int]
        Unique-vertex IDs in `analysis` for every vertex shared by
        the two graphs.
    context : NDArray[bool]
        Mask over the shared vertices selecting those that may
        participate in a shared run.

    Returns
    -------
    prev_alsos : NDArray[bool]
        Whether each shared vertex is immediately preceded in an arc
        by another context vertex.
    after_alsos : NDArray[bool]
        Whether each shared vertex is immediately followed in an arc
        by another context vertex.

    Notes
    -----
    Adjacency is evaluated only between consecutive rows belonging
    to the same arc. Integer lookups avoid repeatedly hashing
    coordinate tuples.
    """
    prev_alsos = np.zeros(shared_vtx_ids.shape[0], dtype=bool)
    after_alsos = np.zeros(shared_vtx_ids.shape[0], dtype=bool)
    if (shared_vtx_ids.size == 0) or (endpts.shape[0] == 0):
        return prev_alsos, after_alsos

    # Translate graph-local unique IDs to positions in the shared arrays.
    unique_to_shared = np.full(analysis.vtxs.shape[0], -1, dtype=np.intp)
    unique_to_shared[shared_vtx_ids] = np.arange(shared_vtx_ids.size)
    vtx_shared_ids = np.full(analysis.vtx_inv_ids.shape, -1, dtype=np.intp)
    used = analysis.vtx_inv_ids >= 0
    vtx_shared_ids[used] = unique_to_shared[analysis.vtx_inv_ids[used]]

    # A difference array marks consecutive stored rows belonging to one arc.
    segment_counts = np.zeros(analysis.vtx_inv_ids.size, dtype=np.int32)
    np.add.at(segment_counts, endpts[:, 0], 1)
    np.add.at(segment_counts, endpts[:, 1], -1)
    same_arc = np.cumsum(segment_counts)[:-1] > 0
    left = vtx_shared_ids[:-1]
    right = vtx_shared_ids[1:]
    valid = same_arc & (left >= 0) & (right >= 0)
    valid_ids = np.flatnonzero(valid)
    if valid_ids.size:
        valid_ids = valid_ids[context[left[valid_ids]] & context[right[valid_ids]]]
        np.logical_or.at(after_alsos, left[valid_ids], True)
        np.logical_or.at(prev_alsos, right[valid_ids], True)
    return prev_alsos, after_alsos


def _split_arcs_at_vertex_ids(
    orders: NDArray[np.integer],
    vtxs: NDArray[NpCoords],
    endpts: NDArray[NpIndex],
    analysis: _GraphVertexAnalysis,
    split_coord_ids: NDArray[NpIndex],
) -> tuple[NDArray[np.integer], NDArray[NpCoords], NDArray[NpIndex]]:
    """
    Splits all requested interior vertices while rebuilding arrays
    once.

    Parameters
    ----------
    orders : NDArray[int]
        Order of every input arc.
    vtxs : NDArray[int | float]
        (V,n) array of stored vertex coordinates.
    endpts : NDArray[int]
        (A,2) array of inclusive arc ranges into `ijs`.
    analysis : _GraphVertexAnalysis
        Reusable metadata previously calculated from `ijs` and
        `endpts`.
    split_coord_ids : NDArray[int]
        Unique-vertex IDs in `analysis` that should become arc
        endpoints.

    Returns
    -------
    orders : NDArray[int]
        Orders of the rebuilt arcs, with the source order copied to
        every split portion.
    vtxs : NDArray[int | float]
        Compact vertex array containing the rebuilt arcs.
    endpts : NDArray[int]
        Inclusive ranges of the rebuilt arcs in the returned vertex
        array.

    Notes
    -----
    Every matching interior occurrence is split. The function
    collects views of all resulting portions and concatenates them
    once, avoiding repeated growth of the vertex, endpoint, and
    order arrays.
    """
    if split_coord_ids.size == 0:
        return orders, vtxs, endpts

    # Convert unique-coordinate IDs to a mask over stored vertex rows
    split_vtxs = np.isin(analysis.vtx_inv_ids, split_coord_ids)
    chunks: list[NDArray[NpCoords]] = []
    new_orders: list[np.generic | int] = []
    lengths: list[int] = []
    for order, (start, end) in zip(orders, endpts):
        # Existing arc endpoints are boundaries already; split only interiors
        interior = np.flatnonzero(split_vtxs[start + 1 : end]) + start + 1
        boundaries = np.concatenate(([start], interior, [end]))
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            chunk = vtxs[left : right + 1]
            chunks.append(chunk)
            lengths.append(chunk.shape[0])
            new_orders.append(order)

    # Materialise the compact graph once and derive its inclusive arc ranges
    new_vtxs = np.concatenate(chunks, axis=0)
    ends = np.cumsum(lengths, dtype=np.intp) - 1
    starts = np.concatenate(([0], ends[:-1] + 1))
    new_endpts = np.column_stack((starts, ends)).astype(endpts.dtype, copy=False)
    return np.asarray(new_orders, dtype=orders.dtype), new_vtxs, new_endpts


def solve_graph_overlaps(
    g1_orders: NDArray[NpInt],
    g1_vtxs: NDArray[NpCoords],
    g1_endpts: NDArray[NpIndex],
    g2_orders: NDArray[NpInt],
    g2_vtxs: NDArray[NpCoords],
    g2_endpts: NDArray[NpIndex],
    allows_arcs_overlap: bool = True,
    remove_unused: bool = False,
) -> tuple[
    NDArray[NpInt],
    NDArray[NpCoords],
    NDArray[NpIndex],
    NDArray[NpInt],
    NDArray[NpCoords],
    NDArray[NpIndex],
]:
    """
    Splits two graphs at shared vertices to align their arc
    endpoints.

    Vertices that are endpoints in only one graph are inserted as
    endpoints in the other graph. Interior overlaps are inserted
    into both graphs unless they belong to a shared arc and
    `allows_arcs_overlap` is `True`.

    Parameters
    ----------
    g1_orders : NDArray[int]
        (A1,) array containing the order of each arc in the first
        graph.
    g1_vtxs : NDArray[int | float]
        (V1,n) array containing the coordinates of all vertices
        stored for the first graph.
    g1_endpts : NDArray[int]
        (A1,2) array containing the inclusive starting and ending
        vertex indices of each arc in the first graph.
    g2_orders : NDArray[int]
        (A2,) array containing the order of each arc in the second
        graph.
    g2_vtxs : NDArray[int | float]
        (V2,n) array containing the coordinates of all vertices
        stored for the second graph.
    g2_endpts : NDArray[int]
        (A2,2) array containing the inclusive starting and ending
        vertex indices of each arc in the second graph.
    allows_arcs_overlap : bool, optional
        Whether shared sequences of interior vertices may remain
        overlapping without being split into separate arcs.
        If true, consecutive overlap of vertices are isolated as a
        new arc, which will be identical between the two input
        graphs (aside form the directivity).
        Default option is `True`.
    remove_unused : bool, optional
        Whether to compact both returned vertex arrays so their arc
        ranges are adjacent.
        Default option is `False`.

    Returns
    -------
    g1_orders : NDArray[int]
        Updated orders of the arcs in the first graph.
    g1_vtxs : NDArray[int | float]
        Updated vertex coordinates of the first graph.
    g1_endpts : NDArray[int]
        Updated inclusive endpoint indices of the arcs in the first
        graph.
    g2_orders : NDArray[int]
        Updated orders of the arcs in the second graph.
    g2_vtxs : NDArray[int | float]
        Updated vertex coordinates of the second graph.
    g2_endpts : NDArray[int]
        Updated inclusive endpoint indices of the arcs in the second
        graph.
    """
    g1_analysis = _analyse_graph_vertices(g1_vtxs, g1_endpts)
    g2_analysis = _analyse_graph_vertices(g2_vtxs, g2_endpts)
    shared = _find_shared_graph_vertices(g1_analysis, g2_analysis)
    g1_ep = shared.g1_is_endpt
    g2_ep = shared.g2_is_endpt
    intr_intr = ~g1_ep & ~g2_ep
    g1_intr_g2_vert = ~g1_ep & g2_ep
    g1_vert_g2_intr = g1_ep & ~g2_ep

    split_both = intr_intr.copy()
    if allows_arcs_overlap and np.any(intr_intr):
        # Coordinates already duplicated across arcs, together with
        # mismatched endpoints that this call will split, bound an
        # existing shared run
        context = intr_intr | (
            (g1_ep & g2_ep & (shared.g1_cnts > 1) & (shared.g2_cnts > 1))
            | (g1_intr_g2_vert & (shared.g2_cnts > 1))
            | (g1_vert_g2_intr & (shared.g1_cnts > 1))
        )
        g1_prev, g1_after = _find_shared_vertex_neighbours(
            g1_analysis, g1_endpts, shared.g1_vtx_ids, context
        )
        g2_prev, g2_after = _find_shared_vertex_neighbours(
            g2_analysis, g2_endpts, shared.g2_vtx_ids, context
        )
        split_both = intr_intr & ~(g1_prev & g1_after & g2_prev & g2_after)

    g1_split = g1_intr_g2_vert | split_both
    g2_split = g1_vert_g2_intr | split_both
    g1_orders, g1_vtxs, g1_endpts = _split_arcs_at_vertex_ids(
        g1_orders,
        g1_vtxs,
        g1_endpts,
        g1_analysis,
        shared.g1_vtx_ids[g1_split],  # type: ignore
    )
    g2_orders, g2_vtxs, g2_endpts = _split_arcs_at_vertex_ids(
        g2_orders,
        g2_vtxs,
        g2_endpts,
        g2_analysis,
        shared.g2_vtx_ids[g2_split],  # type: ignore
    )
    if remove_unused:
        g1_vtxs, g1_endpts = remove_unused_vertices(g1_vtxs, g1_endpts)
        g2_vtxs, g2_endpts = remove_unused_vertices(g2_vtxs, g2_endpts)
    return g1_orders, g1_vtxs, g1_endpts, g2_orders, g2_vtxs, g2_endpts


def _resolve_topology_intersections(
    vtxs: NDArray[NpCoords],
    endpts: NDArray[NpIndex],
    vtx_keeps: NDArray[np.bool_],
    tol: float,
    graph_ids: Optional[NDArray[np.integer]] = None,
    max_iters: int = 4,
) -> NDArray[np.bool_]:
    """
    Checks for topology violations and resolves them by iteratively
    reducing the tolerance for conflicting arcs and re-simplifying
    them.

    Parameters
    ----------
    vtxs : NDArray[int | float]
    endpts : NDArray[int]
    vtx_keeps : NDArray[bool]
    tol : float
    graph_ids : NDArray[int] | None
    max_iters : int
        - max_iters = 0: Validate, but make no repair attempts.
        - max_iters = 1: Make at most one repair attempt.
        - max_iters = N: Make at most N attempts.
    """
    vertex_cumsum = np.cumsum(vtx_keeps) - 1
    vertices_aux = vtxs[:, vtx_keeps]
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
            vtx_keeps[start : end + 1] = simp_f.simplify_flowgraph(
                vtxs[:, start : end + 1].astype(np.float32, order="F"),
                np.array([[1], [arc_length]], dtype=np.int32, order="F"),
                tol,
            ).astype(bool)
        # Squeeze the vertices and map the arc endpoints to the new indices
        vertex_cumsum = np.cumsum(vtx_keeps) - 1
        vertices_aux = vtxs[:, vtx_keeps]
        endpts_aux = vertex_cumsum[endpts]

        intxs = _locate_disallowed_graph_topology(vertices_aux, endpts_aux, graph_ids)
    # If there are still intersections after that many iterations, don't simplify those arc
    if intxs is not None:
        for iarc in np.unique(intxs[:, :2]):
            vtx_keeps[endpts[0, iarc] : endpts[1, iarc] + 1] = True
    return vtx_keeps.astype(bool)
