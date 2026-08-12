"""
Triangulates 2D points using the Python backend.

This internal module implements incremental Bowyer-Watson
triangulation for the public meshing API.

Created: 2026-08-12, En-Chi Lee (williameclee@gmail.com)
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

    minx = int(np.min(vtxs[:, 0]))
    maxx = int(np.max(vtxs[:, 0]))
    miny = int(np.min(vtxs[:, 1]))
    maxy = int(np.max(vtxs[:, 1]))
    span = max(maxx - minx, maxy - miny, 1)
    midx = minx + (maxx - minx) // 2
    midy = miny + (maxy - miny) // 2

    super_vtxs = np.array(
        [
            (midx - 4 * span, midy - 2 * span),
            (midx + 4 * span, midy - 2 * span),
            (midx, midy + 4 * span),
        ],
        dtype=vtxs.dtype,
    )

    supertriangle = (
        int(vtxs.shape[0]),
        int(vtxs.shape[0] + 1),
        int(vtxs.shape[0] + 2),
    )

    return (np.vstack((vtxs, super_vtxs)), supertriangle)


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
