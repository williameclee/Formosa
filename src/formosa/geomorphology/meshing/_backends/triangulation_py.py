"""
Triangulates 2D points using the Python backend.

This internal module implements incremental Bowyer-Watson
triangulation for the public meshing API.

Created: 2026-08-12, En-Chi Lee (williameclee@gmail.com)
Last modified: 2026-08-13, En-Chi Lee (williameclee@gmail.com)
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
