"""
Triangulates 2D points using the Python backend.

This internal module implements incremental Bowyer-Watson
triangulation for the public meshing API.

Created: 2026-08-12, En-Chi Lee (williameclee@gmail.com)
Last modified: 2026-08-13, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np
from collections import Counter

from formosa.geomorphology.drainage.network import GraphTopologyError
from formosa.geomorphology.geometry import incircle, orient_v2

from numpy.typing import NDArray
from formosa.utils.typing import NpCoords, NpCanonIndex


def add_supertriangle(
    vtxs: NDArray[NpCoords],
) -> tuple[NDArray[NpCoords], tuple[int, int, int]]:
    """
    Appends a super-triangle."""

    if vtxs.ndim != 2 or vtxs.shape[1] != 2:
        raise ValueError("Vertices must have shape (V, 2), " + f"but got {vtxs.shape}.")
    if vtxs.shape[0] == 0:
        raise ValueError("At least one point is required.")

    if np.issubdtype(vtxs.dtype, np.integer):
        all_vtxs = vtxs.astype(np.int64, copy=False)
    else:
        all_vtxs = vtxs

    minx = int(np.min(all_vtxs[:, 0]))
    maxx = int(np.max(all_vtxs[:, 0]))
    miny = int(np.min(all_vtxs[:, 1]))
    maxy = int(np.max(all_vtxs[:, 1]))
    xspan = max(maxx - minx, 1)
    yspan = max(maxy - miny, 1)
    midx = minx + (maxx - minx) // 2

    super_vtxs = np.array(
        [
            (midx - 3 * xspan, miny - yspan),
            (midx + 3 * xspan, miny - yspan),
            (midx, maxy + 2 * yspan),
        ],
        dtype=all_vtxs.dtype,
    )

    supertriangle = (
        int(vtxs.shape[0]),
        int(vtxs.shape[0] + 1),
        int(vtxs.shape[0] + 2),
    )

    return (np.vstack((all_vtxs, super_vtxs)), supertriangle)  # type: ignore


def edge_key(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


def insert_vertex(
    vtx_id: int,
    vtxs: np.ndarray,
    triangles: list[tuple[int, int, int]],
) -> list[tuple[int, int, int]]:
    """
    Inserts a new vertex into a Delaunay triangulation using Bowyer-
    Watson algorithm.
    """

    bad_tri_ids = []

    # Find all triangles whose circumcircle contain the new point
    for tri_id, (a, b, c) in enumerate(triangles):
        det = incircle(
            vtxs[a],
            vtxs[b],
            vtxs[c],
            vtxs[vtx_id],
            oriented=True,
            backend="python",
        )

        if det > 0:
            bad_tri_ids.append(tri_id)

    if not bad_tri_ids:
        raise GraphTopologyError(
            f"Point {vtx_id} does not lie in any triangulation cavity."
        )

    edge_cnts: Counter[tuple[int, int]] = Counter()

    for tri_id in bad_tri_ids:
        a, b, c = triangles[tri_id]
        edge_cnts.update(
            (
                edge_key(a, b),
                edge_key(b, c),
                edge_key(c, a),
            )
        )

    cavity_edges = [edge for edge, cnt in edge_cnts.items() if cnt == 1]

    bad_tri_ids = set(bad_tri_ids)

    new_triangles = [
        triangle
        for triangle_id, triangle in enumerate(triangles)
        if triangle_id not in bad_tri_ids
    ]

    for u, v in cavity_edges:
        candidate = (u, v, vtx_id)

        orient = orient_v2(vtxs[u], vtxs[v], vtxs[vtx_id], backend="python")
        if orient > 0:
            new_triangles.append(candidate)
        elif orient < 0:
            new_triangles.append((v, u, vtx_id))
        else:
            raise GraphTopologyError(
                f"Cavity edge {(u, v)} and point {vtx_id} are collinear."
            )

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
    orientation = orient_v2(
        vtxs[a],
        vtxs[b],
        vtxs[c],
        backend="python",
    )

    if orientation > 0:
        return triangle
    if orientation < 0:
        return (a, c, b)

    raise GraphTopologyError(f"Degenerate triangle {triangle} has collinear vertices.")


def _validate_triangulate_points(points: NDArray[NpCoords]) -> None:
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Points must have shape (P, 2), " + f"but got {points.shape}.")
    if points.shape[0] < 3:
        raise ValueError(
            "At least three points are required, " + f"but only got {points.shape[0]}."
        )
    n_unq_pts = np.unique(points, axis=0).shape[0]
    if n_unq_pts != points.shape[0]:
        raise ValueError(
            "Points must be unique, "
            + f"but found {points.shape[0]-n_unq_pts} duplicates."
        )


def triangulate_points(vtxs: NDArray[NpCoords]) -> NDArray[NpCanonIndex]:
    vtxs = np.asarray(vtxs)
    _validate_triangulate_points(vtxs)

    n_pts = vtxs.shape[0]

    vtxs, supertriangle = add_supertriangle(vtxs)
    triangles = [order_ccw(supertriangle, vtxs)]

    # Vertex IDs stay unchanged. Only insertion order changes.
    for point_id in range(n_pts):
        triangles = insert_vertex(
            point_id,
            vtxs,
            triangles,
        )

    # Remove triangles connected to temporary supertriangle vertices.
    triangles = [
        triangle
        for triangle in triangles
        if all(vertex_id < n_pts for vertex_id in triangle)
    ]

    if not triangles:
        raise GraphTopologyError("Point set did not produce any finite triangles.")

    return np.asarray(triangles, dtype=NpCanonIndex)


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
