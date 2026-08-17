"""
Performs unconstrained 2D triangulation using the Python backend.

This internal module implements incremental Bowyer-Watson
triangulation and facet-neighbour construction for the public
meshing API. Constrained edge recovery is implemented separately in
`constrained_triangulation_py.py`.

Created: 2026-08-12, En-Chi Lee (williameclee@gmail.com)
Last modified: 2026-08-17, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np

from formosa.geomorphology.drainage.network import GraphTopologyError
from formosa.geomorphology.geometry import incircle, orient

from numpy.typing import NDArray
from formosa.utils.typing import NpCoords, NpCanonIndex


def _canonical_edge(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


def make_initial_facets(
    vtxs: NDArray[NpCoords],
) -> tuple[list[tuple[int, int, int]], set[int], int]:
    """
    Builds one finite seed face and its three symbolic infinite
    faces.

    Parameters
    ----------
    vtxs : NDArray[int]
        Unique vertex coordinate indices.

    Returns
    -------
    faces : list[tuple[int, int, int]]
        Initial triangulation facets.
        Each entry contains the index of the 3 vertices that form
        the triangular facet.
    seeds : set[int]
    infinite_id : int
        Index that represents the infinite vertex in `faces`.
    """
    # Select first 2 vertices (input-order independent), and a third
    # to form the first triangle
    order = np.lexsort((vtxs[:, 1], vtxs[:, 0]))
    a = int(order[0])
    b = int(order[1])
    c = -1
    for candidate in order[2:]:
        candidate = int(candidate)
        if orient(vtxs[a], vtxs[b], vtxs[candidate], backend="python") != 0:
            c = candidate
            break
    # Fails if all other vertices are collinear with the first 2
    # vertices
    if c < 0:
        raise GraphTopologyError("Point set is collinear.")

    face = order_ccw((a, b, c), vtxs)
    a, b, c = face
    iinf = int(vtxs.shape[0])
    faces = [face, (iinf, b, a), (iinf, c, b), (iinf, a, c)]
    return faces, {a, b, c}, iinf


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


def is_bad_facet(
    face: tuple[int, int, int],
    ivtx: int,
    vtxs: NDArray[NpCoords],
    iinf: int,
) -> bool:
    """
    Classifies a finite point against a finite or infinite facet.
    """
    if ivtx == iinf:
        raise GraphTopologyError("The infinite vertex cannot be inserted.")

    ninf = face.count(iinf)
    if ninf > 1:
        raise GraphTopologyError(
            f"Facet must contains no more than 1 infinite vertex, "
            + f"but {face} got {ninf}."
        )

    # No infinite vertex:
    # Is bad if new vertex is in the circumcircle
    if ninf == 0:
        a, b, c = face
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
    i = face.index(iinf)
    u = face[(i + 1) % 3]
    v = face[(i + 2) % 3]
    o = orient(vtxs[u], vtxs[v], vtxs[ivtx], backend="python")
    return o > 0 or (o == 0 and _point_on_segment(vtxs[u], vtxs[v], vtxs[ivtx]))


def insert_vertex(
    ivtx: int,
    vtxs: NDArray[NpCoords],
    faces: list[tuple[int, int, int]],
    iinf: int,
) -> list[tuple[int, int, int]]:
    """
    Inserts a new vertex into a Delaunay triangulation using Bowyer-
    Watson algorithm.
    """

    # Find all triangles whose circumcircle contain the new point
    bad_faces = [
        iface
        for iface, face in enumerate(faces)
        if is_bad_facet(face, ivtx, vtxs, iinf)
    ]

    if not bad_faces:
        raise GraphTopologyError(
            f"Point {ivtx} does not lie in any triangulation cavity."
        )

    # Keep the direction of cavity-boundary edges. Adjacent oriented
    # facets traverse their shared edge oppositely, so interior
    # edges cancel. The retained direction also orients new symbolic
    # infinite faces without a geometric predicate involving the
    # infinite vertex.
    edges: dict[tuple[int, int], tuple[int, int]] = {}

    for iface in bad_faces:
        a, b, c = faces[iface]
        for u, v in ((a, b), (b, c), (c, a)):
            key = _canonical_edge(u, v)
            prev = edges.pop(key, None)
            if prev is not None and prev != (v, u):
                raise GraphTopologyError(
                    f"Facets incident to cavity edge {key} have inconsistent orientation."
                )
            if prev is None:
                edges[key] = (u, v)

    bad_faces = set(bad_faces)

    new_faces = [face for iface, face in enumerate(faces) if iface not in bad_faces]

    for u, v in edges.values():
        candidate = (u, v, ivtx)
        if iinf not in candidate:
            o = orient(vtxs[u], vtxs[v], vtxs[ivtx], backend="python")
            if o <= 0:
                raise GraphTopologyError(
                    f"Cavity edge {(u, v)} and point {ivtx} do not form "
                    + "a counterclockwise finite face."
                )
        new_faces.append(candidate)

    return new_faces


def order_ccw(
    face: tuple[int, int, int],
    vtxs: NDArray[NpCoords],
) -> tuple[int, int, int]:
    """
    Reorders vertices of a triangle such that it is
    counterclockwise.
    """

    a, b, c = face
    o = orient(vtxs[a], vtxs[b], vtxs[c], backend="python")

    if o > 0:
        return face
    if o < 0:
        return (a, c, b)

    raise GraphTopologyError(f"Degenerate triangle {face} has collinear vertices.")


def triangulate_points(vtxs: NDArray[NpCoords]) -> NDArray[NpCanonIndex]:
    faces, seed_ids, iinf = make_initial_facets(vtxs)

    # Add vertices by a deterministic order independent of input
    insertion_order = np.lexsort((vtxs[:, 1], vtxs[:, 0]))
    for ivtx in map(int, insertion_order):
        if ivtx in seed_ids:
            continue
        faces = insert_vertex(ivtx, vtxs, faces, iinf)

    # Remove symbolic infinite faces
    val_faces = [face for face in faces if iinf not in face]

    if not val_faces:
        raise GraphTopologyError(
            "Point set did not produce any finite triangle facets; "
            + f"all facets are incident to infinite vertex {iinf}.\n"
            + f"{faces}"
        )

    return np.asarray(val_faces, dtype=NpCanonIndex)


def find_facet_neighbours(
    faces: NDArray[NpCanonIndex],
) -> tuple[NDArray[NpCanonIndex], dict[tuple[int, int], tuple[int, int]]]:
    """
    Builds facet-to-facet adjacency and an edge lookup.

    Returns
    -------
    nabrs : NDArray[int32]
        `nabrs[iface, side]` is the triangle across that side,
        or `-1` at the mesh boundary.
    edge_owners : dict[tuple[int, int], tuple[int, int]]
        Maps a canonical edge to one incident `(triangle, side)`.
    """
    faces = np.asarray(faces)

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(
            "Triangle facets must have shape (F, 3), " + f"but got {faces.shape}."
        )
    if np.any(faces < 0):
        raise ValueError("Triangle vertex IDs must be non-negative.")

    nfaces = faces.shape[0]
    nabrs = np.full((nfaces, 3), -1, dtype=np.int32)

    # key -> (triangle ID, side ID, directed start)
    owners: dict[tuple[int, int], tuple[int, int, int]] = {}

    for iface, (a, b, c) in enumerate(faces):
        if a == b or b == c or c == a:
            raise ValueError(f"Triangle facet {iface} is degenerate.")

        for iside, (u, v) in enumerate(((b, c), (c, a), (a, b))):
            u = int(u)
            v = int(v)
            key = (u, v) if u < v else (v, u)

            prev = owners.get(key)
            if prev is None:
                owners[key] = (iface, iside, u)
                continue

            jface, jside, other_start = prev

            if nabrs[jface, jside] != -1:
                raise ValueError(f"Edge {key} belongs to more than 2 triangle facets.")

            # Adjacent CCW triangles must traverse their shared edge
            # in opposite directions.
            if other_start == u:
                raise ValueError(
                    f"Triangles incident to edge {key} have inconsistent orientation."
                )

            nabrs[iface, iside] = jface
            nabrs[jface, jside] = iface

    edge_owners = {edge: (iface, iside) for edge, (iface, iside, _) in owners.items()}
    return nabrs, edge_owners
