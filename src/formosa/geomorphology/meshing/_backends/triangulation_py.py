"""
Triangulates 2D points using the Python backend.

This internal module implements incremental Bowyer-Watson
triangulation for the public meshing API.

Created: 2026-08-12, En-Chi Lee (williameclee@gmail.com)
Last modified: 2026-08-16, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np

from formosa.geomorphology.drainage.network import GraphTopologyError
from formosa.geomorphology.geometry import incircle, orient_v2

from numpy.typing import NDArray
from formosa.utils.typing import NpCoords, NpCanonIndex


def edge_key(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


def make_initial_facets(
    vtxs: NDArray[NpCoords],
) -> tuple[list[tuple[int, int, int]], set[int], int]:
    """
    Builds one finite seed face and its three symbolic infinite faces.

    Parameters
    ----------
    vtxs : NDArray[int]
        Unique vertex coordinate indices.

    Returns
    -------
    triangles : list[tuple[int, int, int]]
        Initial triangulation facets.
        Each entry contains the index of the 3 vertices that form
        the triangular facet.
    ids : set[int]
    infinite_id : int
        Index that represents the infinite vertex in `triangles`.
    """
    # Select first 2 vertices (input-order independent), and a third
    # to form the first triangle
    order = np.lexsort((vtxs[:, 1], vtxs[:, 0]))
    a = int(order[0])
    b = int(order[1])
    c = -1
    for candidate in order[2:]:
        candidate = int(candidate)
        if orient_v2(vtxs[a], vtxs[b], vtxs[candidate], backend="python") != 0:
            c = candidate
            break
    # Fails if all other vertices are collinear with the first 2
    # vertices
    if c < 0:
        raise GraphTopologyError("Point set is collinear.")

    tri = order_ccw((a, b, c), vtxs)
    a, b, c = tri
    iinf = int(vtxs.shape[0])
    triangles = [tri, (iinf, b, a), (iinf, c, b), (iinf, a, c)]
    return triangles, {a, b, c}, iinf


def _point_on_segment(
    a: NDArray[NpCoords], b: NDArray[NpCoords], p: NDArray[NpCoords]
) -> bool:
    """
    Returns whether collinear `p` lies on the closed segment `a-b`.
    """
    return bool(
        min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
        and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])
    )


def is_bad_triangle(
    triangle: tuple[int, int, int],
    ivtx: int,
    vtxs: NDArray[NpCoords],
    iinf: int,
) -> bool:
    """
    Classifies a finite point against a finite or infinite facet.
    """
    if ivtx == iinf:
        raise GraphTopologyError("The infinite vertex cannot be inserted.")

    inf_cnt = triangle.count(iinf)
    if inf_cnt > 1:
        raise GraphTopologyError(
            f"Facet must contains no more than 1 infinite vertex, "
            + f"but {triangle} got {inf_cnt}."
        )

    # No infinite vertex:
    # Is bad if new vertex is in the circumcircle
    if inf_cnt == 0:
        a, b, c = triangle
        return (
            incircle(
                vtxs[a], vtxs[b], vtxs[c], vtxs[ivtx], oriented=True, backend="python"
            )
            > 0
        )

    # Infinite facets are stored as cyclically oriented triples. If
    # the infinite vertex occurs at i, u-v is the finite edge as
    # traversed by that facet, opposite to the CCW convex-hull
    # direction. A point sees the hull edge when it lies to the left
    # of u-v. A point on the hull segment is included as well so
    # insertion splits both incident facets.
    i = triangle.index(iinf)
    u = triangle[(i + 1) % 3]
    v = triangle[(i + 2) % 3]
    orient = orient_v2(vtxs[u], vtxs[v], vtxs[ivtx], backend="python")
    return orient > 0 or (
        orient == 0 and _point_on_segment(vtxs[u], vtxs[v], vtxs[ivtx])
    )


def insert_vertex(
    ivtx: int,
    vtxs: NDArray[NpCoords],
    triangles: list[tuple[int, int, int]],
    iinf: int,
) -> list[tuple[int, int, int]]:
    """
    Inserts a new vertex into a Delaunay triangulation using Bowyer-
    Watson algorithm.
    """

    # Find all triangles whose circumcircle contain the new point
    bad_tri_ids = [
        itri
        for itri, tri in enumerate(triangles)
        if is_bad_triangle(tri, ivtx, vtxs, iinf)
    ]

    if not bad_tri_ids:
        raise GraphTopologyError(
            f"Point {ivtx} does not lie in any triangulation cavity."
        )

    # Keep the direction of cavity-boundary edges. Adjacent oriented
    # facets traverse their shared edge oppositely, so interior
    # edges cancel. The retained direction also orients new symbolic
    # infinite faces without a geometric predicate involving the
    # infinite vertex.
    edges: dict[tuple[int, int], tuple[int, int]] = {}

    for tri_id in bad_tri_ids:
        a, b, c = triangles[tri_id]
        for u, v in ((a, b), (b, c), (c, a)):
            key = edge_key(u, v)
            prev = edges.pop(key, None)
            if prev is not None and prev != (v, u):
                raise GraphTopologyError(
                    f"Faces incident to cavity edge {key} have inconsistent orientation."
                )
            if prev is None:
                edges[key] = (u, v)

    bad_tri_ids = set(bad_tri_ids)

    new_triangles = [
        triangle
        for triangle_id, triangle in enumerate(triangles)
        if triangle_id not in bad_tri_ids
    ]

    for u, v in edges.values():
        candidate = (u, v, ivtx)
        if iinf not in candidate:
            orient = orient_v2(vtxs[u], vtxs[v], vtxs[ivtx], backend="python")
            if orient <= 0:
                raise GraphTopologyError(
                    f"Cavity edge {(u, v)} and point {ivtx} do not form "
                    + "a counterclockwise finite face."
                )
        new_triangles.append(candidate)

    return new_triangles


def order_ccw(
    triangle: tuple[int, int, int],
    vtxs: NDArray[NpCoords],
) -> tuple[int, int, int]:
    """
    Reorders vertices of a triangle such that it is
    counterclockwise.
    """

    a, b, c = triangle
    orient = orient_v2(vtxs[a], vtxs[b], vtxs[c], backend="python")

    if orient > 0:
        return triangle
    if orient < 0:
        return (a, c, b)

    raise GraphTopologyError(f"Degenerate triangle {triangle} has collinear vertices.")


def triangulate_points(vtxs: NDArray[NpCoords]) -> NDArray[NpCanonIndex]:
    triangles, seed_ids, iinf = make_initial_facets(vtxs)

    # Add vertices by a deterministic order independent of input
    insertion_order = np.lexsort((vtxs[:, 1], vtxs[:, 0]))
    for ivtx in map(int, insertion_order):
        if ivtx in seed_ids:
            continue
        triangles = insert_vertex(ivtx, vtxs, triangles, iinf)

    # Remove symbolic infinite faces
    val_triangles = [triangle for triangle in triangles if iinf not in triangle]

    if not val_triangles:
        raise GraphTopologyError(
            "Point set did not produce any finite triangles; "
            + f"all triangles are incident to infinite vertex {iinf}.\n"
            + f"{triangles}"
        )

    return np.asarray(val_triangles, dtype=NpCanonIndex)


def find_triangle_neighbours(
    triangles: NDArray[NpCanonIndex],
) -> tuple[
    NDArray[NpCanonIndex],
    dict[tuple[int, int], tuple[int, int]],
]:
    """
    Builds triangle-to-triangle adjacency and an edge lookup.

    Returns
    -------
    neighbours : NDArray[int32]
        `neighbours[itri, side]` is the triangle across that side,
        or `-1` at the mesh boundary.
    edge_owners : dict[tuple[int, int], tuple[int, int]]
        Maps a canonical edge to one incident `(triangle, side)`.
    """
    triangles = np.asarray(triangles)

    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("Triangles must have shape (F, 3).")
    if np.any(triangles < 0):
        raise ValueError("Triangle vertex IDs must be non-negative.")

    ntris = triangles.shape[0]
    neighbours = np.full((ntris, 3), -1, dtype=np.int32)

    # key -> (triangle ID, side ID, directed start)
    owners: dict[tuple[int, int], tuple[int, int, int]] = {}

    for itri, (a, b, c) in enumerate(triangles):
        if a == b or b == c or c == a:
            raise ValueError(f"Triangle {itri} is degenerate.")

        for iside, (u, v) in enumerate(((b, c), (c, a), (a, b))):
            u = int(u)
            v = int(v)
            key = (u, v) if u < v else (v, u)

            previous = owners.get(key)
            if previous is None:
                owners[key] = (itri, iside, u)
                continue

            jtri, jside, other_start = previous

            if neighbours[jtri, jside] != -1:
                raise ValueError(f"Edge {key} belongs to more than two triangles.")

            # Adjacent CCW triangles must traverse their shared edge
            # in opposite directions.
            if other_start == u:
                raise ValueError(
                    f"Triangles incident to edge {key} have inconsistent orientation."
                )

            neighbours[itri, iside] = jtri
            neighbours[jtri, jside] = itri

    edge_owners = {
        edge: (itri, iside) for edge, (itri, iside, _) in owners.items()
    }
    return neighbours, edge_owners


def _update_flipped_neighbours(
    nabrs: NDArray[NpCanonIndex], itri: int, iside: int, jtri: int, jside: int
) -> NDArray[NpCanonIndex]:
    f_nabrs = np.array(nabrs, dtype=NpCanonIndex, order="C", copy=True)
    inabrs = nabrs[itri]
    jnabrs = nabrs[jtri]
    f_nabrs[itri] = (jnabrs[(jside + 1) % 3], jtri, inabrs[(iside + 2) % 3])
    f_nabrs[jtri] = (jnabrs[(jside + 2) % 3], inabrs[(iside + 1) % 3], itri)
    moved_to_i = int(jnabrs[(jside + 1) % 3])
    if moved_to_i != -1:
        f_nabrs[moved_to_i][f_nabrs[moved_to_i] == jtri] = itri

    moved_to_j = int(inabrs[(iside + 1) % 3])
    if moved_to_j != -1:
        f_nabrs[moved_to_j][f_nabrs[moved_to_j] == itri] = jtri
    return f_nabrs


def flip_triangle_edge(
    vtxs: NDArray[NpCoords],
    triangles: NDArray[NpCanonIndex],
    nabrs: NDArray[NpCanonIndex],
    itri: int,
    iside: int,
) -> tuple[NDArray[NpCanonIndex], NDArray[NpCanonIndex]]:
    """
    Flips one interior triangle edge in a convex quadrilateral.

    The input arrays are not modified. Neighbours are recomputed
    after the flip to keep this initial implementation simple.
    """
    if itri < 0 or itri >= triangles.shape[0]:
        raise IndexError(f"Triangle ID {itri} is out of bounds.")
    elif iside < 0 or iside >= 3:
        raise IndexError(f"Triangle side ID {iside} is out of bounds.")

    jtri = int(nabrs[itri, iside])
    if jtri == -1:
        raise GraphTopologyError("A boundary edge cannot be flipped.")

    # Find which corresponding side is the edge to the other traingle
    reciprocal_sides = np.flatnonzero(nabrs[jtri] == itri)
    if reciprocal_sides.size != 1:
        raise GraphTopologyError(
            f"Triangles {itri} and {jtri} do not have reciprocal neighbours."
        )
    jside = int(reciprocal_sides[0])

    p = int(triangles[itri, iside])  # Not-edge vertex
    u = int(triangles[itri, (iside + 1) % 3])  # Edge vertex 1
    v = int(triangles[itri, (iside + 2) % 3])  # Edge vertex 2
    if not ({u, v} <= set(map(int, triangles[jtri]))):
        raise GraphTopologyError(
            f"Neighbouring triangles {itri} and {jtri} do not share one edge."
        )
    q = int(triangles[jtri, jside])  # Not edge vertex of the other triangle

    if len({p, q, u, v}) != 4:
        raise GraphTopologyError("The incident triangles do not form a quadrilateral.")

    orient_u = orient_v2(vtxs[p], vtxs[q], vtxs[u], backend="python")
    orient_v = orient_v2(vtxs[p], vtxs[q], vtxs[v], backend="python")
    if orient_u == 0 or orient_v == 0:
        raise GraphTopologyError("The flipped edge would create a degenerate triangle.")
    if (orient_u > 0) == (orient_v > 0):
        raise GraphTopologyError(
            "The incident triangles do not form a convex quadrilateral."
        )

    f_triangles = np.array(triangles, dtype=NpCanonIndex, order="C", copy=True)
    f_triangles[itri] = (p, u, q)
    f_triangles[jtri] = (p, q, v)
    f_nabrs = _update_flipped_neighbours(
        nabrs, itri, iside, jtri, jside
    )
    return f_triangles, f_nabrs


def _find_crossing_edges(
    vtxs: NDArray[NpCoords],
    triangles: NDArray[NpCanonIndex],
    nabrs: NDArray[NpCanonIndex],
    edge: tuple[int, int],
) -> list[tuple[int, int, tuple[int, int]]]:
    """
    Finds unique interior edges properly crossing a constraint edge.

    Parameters
    ----------
    vtxs : NDArray[NpCoords]
        Vertex coordinates of shape `(N, 2)`.
    triangles : NDArray[NpCanonIndex]
        Triangle vertex index matrix of shape `(F, 3)`.
    nabrs : NDArray[NpCanonIndex]
        Triangle neighbour index matrix of shape `(F, 3)`.
    edge : tuple[int, int]
        Endpoint indices `(u, v)` defining the target constraint
        line segment.

    Returns
    -------
    crossings : list[tuple[int, int, tuple[int, int]]]
        List of crossing mesh edge descriptors. Each entry is a
        tuple containing:
        - `itri`: Triangle index owning the edge.
        - `iside`: Local side index (0, 1, or 2) corresponding to
            the edge.
        - `key`: Canonical edge representation `(a, b)` with
            `a < b`.
    """
    # Extract unique interior edges as (triangle, side) pairs,
    # selecting the smaller triangle ID to avoid double counting
    ntris = triangles.shape[0]
    itris = np.repeat(np.arange(ntris), 3)
    isides = np.tile(np.arange(3), ntris)
    flat_nabrs = nabrs.ravel()
    unq_intr_edges = (flat_nabrs >= 0) & (itris < flat_nabrs)
    itris = itris[unq_intr_edges]
    isides = isides[unq_intr_edges]

    # Gather endpoint indices and coordinates for each candidate mesh edge
    a_vtx_ids = triangles[itris, (isides + 1) % 3]
    b_vtx_ids = triangles[itris, (isides + 2) % 3]
    vtxs_ = np.asarray(vtxs, dtype=np.result_type(vtxs.dtype, np.int64))
    u_coord: NDArray[NpCoords] = vtxs_[edge[0]]
    v_coord: NDArray[NpCoords] = vtxs_[edge[1]]
    a_coord: NDArray[NpCoords] = vtxs_[triangles[itris, (isides + 1) % 3]]
    b_coord: NDArray[NpCoords] = vtxs_[triangles[itris, (isides + 2) % 3]]

    # Compute orientation of mesh edge endpoints relative to constraint vector
    uv_coord = v_coord - u_coord
    ua_coord = a_coord - u_coord
    ub_coord = b_coord - u_coord
    orient_uva: NDArray[NpCoords] = (
        uv_coord[0] * ua_coord[:, 1] - uv_coord[1] * ua_coord[:, 0]
    )
    orient_uvb: NDArray[NpCoords] = (
        uv_coord[0] * ub_coord[:, 1] - uv_coord[1] * ub_coord[:, 0]
    )

    # Compute orientation of constraint endpoints relative to mesh edge vectors
    ab_coord = b_coord - a_coord
    au_coord = u_coord - a_coord
    av_coord = v_coord - a_coord
    orient_abu = ab_coord[:, 0] * au_coord[:, 1] - ab_coord[:, 1] * au_coord[:, 0]
    orient_abv = ab_coord[:, 0] * av_coord[:, 1] - ab_coord[:, 1] * av_coord[:, 0]

    # Classify proper line-segment crossings with strict opposite orientations
    crossing: NDArray[np.bool_] = (
        ((orient_uva != 0) & (orient_uvb != 0))  # Non-collinear endpoints
        & ((orient_abu != 0) & (orient_abv != 0))  # Non-collinear endpoints
        & ((orient_uva > 0) != (orient_uvb > 0))
        & ((orient_abu > 0) != (orient_abv > 0))
    )

    # Pack crossing edge descriptors into output list
    return [
        (int(itri), int(iside), edge_key(int(a), int(b)))
        for itri, iside, a, b in zip(
            itris[crossing], isides[crossing], a_vtx_ids[crossing], b_vtx_ids[crossing]
        )
    ]


def _mesh_contains_edge(
    triangles: NDArray[NpCanonIndex], edge: tuple[int, int]
) -> bool:
    """
    Checks whether a mesh contains a specified edge.

    Parameters
    ----------
    triangles : NDArray[NpCanonIndex]
        Triangle vertex index matrix of shape (F, 3).
    edge : tuple[int, int]
        Endpoint indices `(u, v)` defining the target edge.

    Returns
    -------
    contains_edge : bool
        `True` if both endpoints co-occur in at least one triangle, `False` otherwise.
    """
    u, v = edge
    return bool(np.any(np.any(triangles == u, axis=1) & np.any(triangles == v, axis=1)))


def _mesh_topology_key(triangles: NDArray[NpCanonIndex]) -> bytes:
    """
    Returns a canonical byte representation of a triangle topology.
    """
    canonical = np.sort(triangles, axis=1)
    order = np.lexsort((canonical[:, 2], canonical[:, 1], canonical[:, 0]))
    return np.ascontiguousarray(canonical[order]).tobytes()


def recover_constraint_edge(
    vtxs: NDArray[NpCoords],
    triangles: NDArray[NpCanonIndex],
    edge: tuple[int, int],
    locked_edges: set[tuple[int, int]] | None = None,
    nabrs: NDArray[NpCanonIndex] | None = None,
) -> tuple[NDArray[NpCanonIndex], NDArray[NpCanonIndex]]:
    """
    Recovers one constraint as a mesh edge using iterative edge
    flips.

    Flips may preserve or reduce, but never increase, the number of
    mesh edges crossing the constraint. Previously visited mesh
    topologies are tracked to prevent infinite flipping loops.
    Existing locked edges are preserved and never flipped. The input arrays are not modified.

    Parameters
    ----------
    vtxs : NDArray[NpCoords]
        Vertex coordinate matrix of shape (N, 2).
    triangles : NDArray[NpCanonIndex]
        Triangle vertex index matrix of shape (F, 3).
    edge : tuple[int, int]
        Endpoint indices `(u, v)` defining the target constraint
        edge.
    locked_edges : set[tuple[int, int]] | None
        Set of canonical edge tuples `(a, b)` with `a < b` that must
        not be flipped.
        Default edges is `None`.
    nabrs : NDArray[NpCanonIndex] | None
        Triangle neighbour matrix of shape (F, 3). If `None`, it is
        computed from `triangles`.
        Default input is `None`.

    Returns
    -------
    recovered_triangles : NDArray[NpCanonIndex]
        Updated triangle vertex index matrix of shape (F, 3).
    updated_neighbours : NDArray[NpCanonIndex]
        Updated triangle neighbour matrix of shape (F, 3).

    Raises
    ------
    ValueError
        If `neighbours` is supplied but its shape does not match
        `triangles`.
    GraphTopologyError
        If constraint edge crosses a locked edge, no flippable edge
        crosses the constraint, or edge flips fail to make progress
        towards edge recovery.
    """
    u, v = map(int, edge)

    target = edge_key(u, v)
    locked = {edge_key(*locked_edge) for locked_edge in (locked_edges or set())}

    r_triangles = np.array(triangles, dtype=NpCanonIndex, order="C", copy=True)
    if nabrs is None:
        nabrs, _ = find_triangle_neighbours(r_triangles)
    else:
        nabrs = np.asarray(nabrs)
        if nabrs.shape != r_triangles.shape:
            raise ValueError(
                "Neighbours must have the same shape as triangles, "
                + f"but got {nabrs.shape} and {r_triangles.shape}."
            )
    if _mesh_contains_edge(r_triangles, target):
        return r_triangles, nabrs

    visited = {_mesh_topology_key(r_triangles)}
    target_recovered = False

    while not target_recovered:
        xngs = _find_crossing_edges(vtxs, r_triangles, nabrs, target)
        if not xngs:
            raise GraphTopologyError(
                f"Constraint edge {target} is absent but crosses no flippable mesh edge."
            )

        locked_xngs = [key for _, _, key in xngs if key in locked]
        if locked_xngs:
            raise GraphTopologyError(
                f"Constraint edge {target} crosses locked mesh edge "
                + f"{locked_xngs[0]}."
            )

        progressed = False
        for itri, iside, _ in xngs:
            try:
                f_triangles, f_nabrs = flip_triangle_edge(
                    vtxs, r_triangles, nabrs, itri, iside
                )
            except GraphTopologyError:
                continue

            f_xngs = _find_crossing_edges(vtxs, f_triangles, f_nabrs, target)
            f_has_target = _mesh_contains_edge(f_triangles, target)
            f_key = _mesh_topology_key(f_triangles)
            if f_key in visited:
                continue
            if not f_has_target and len(f_xngs) > len(xngs):
                continue

            r_triangles = f_triangles
            nabrs = f_nabrs
            target_recovered = f_has_target
            visited.add(f_key)
            progressed = True
            break

        if not progressed:
            raise GraphTopologyError(
                f"No legal edge flip makes progress recovering constraint {target}."
            )

    return r_triangles, nabrs

