"""
Solutions of invalid (overlapping or crossing) flow graph
topologies.

Last modified: 2026-08-03, En-Chi Lee (williameclee@gmail.com)
"""

from dataclasses import dataclass
import numpy as np

from formosa.geomorphology.drainage.network.validation import (
    _locate_disallowed_graph_topology,
)
from formosa.geomorphology.drainage.network.editing import remove_unused_vertices
from formosa.geomorphology._native import network_simplification as simp_f

import numpy.typing as npt
from typing import Optional, TypeVar

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
            vertex_keeps[start : end + 1] = simp_f.simplify_flowgraph(
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
