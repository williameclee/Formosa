"""
Triangulates 2D points using the Python backend.

This internal module implements incremental Bowyer-Watson
triangulation for the public meshing API.

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
) -> tuple[
    NDArray[NpCanonIndex],
    dict[tuple[int, int], tuple[int, int]],
]:
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


def _update_flipped_neighbours(
    nabrs: NDArray[NpCanonIndex], iface: int, iside: int, jface: int, jside: int
) -> NDArray[NpCanonIndex]:
    f_nabrs = np.array(nabrs, dtype=NpCanonIndex, order="C", copy=True)
    inabrs = nabrs[iface]
    jnabrs = nabrs[jface]
    f_nabrs[iface] = (jnabrs[(jside + 1) % 3], jface, inabrs[(iside + 2) % 3])
    f_nabrs[jface] = (jnabrs[(jside + 2) % 3], inabrs[(iside + 1) % 3], iface)
    moved_to_i = int(jnabrs[(jside + 1) % 3])
    if moved_to_i != -1:
        f_nabrs[moved_to_i][f_nabrs[moved_to_i] == jface] = iface

    moved_to_j = int(inabrs[(iside + 1) % 3])
    if moved_to_j != -1:
        f_nabrs[moved_to_j][f_nabrs[moved_to_j] == iface] = jface
    return f_nabrs


def flip_quadrilateral_edge(
    vtxs: NDArray[NpCoords],
    faces: NDArray[NpCanonIndex],
    nabrs: NDArray[NpCanonIndex],
    iface: int,
    iside: int,
) -> tuple[NDArray[NpCanonIndex], NDArray[NpCanonIndex]]:
    """
    Flips an interior triangle edge in a convex quadrilateral.

    The input arrays are not modified.
    """
    if iface < 0 or iface >= faces.shape[0]:
        raise IndexError(f"Triangle ID {iface} is out of bounds.")
    elif iside < 0 or iside >= 3:
        raise IndexError(f"Triangle side ID {iside} is out of bounds.")

    jface = int(nabrs[iface, iside])
    if jface == -1:
        raise GraphTopologyError("A boundary edge cannot be flipped.")

    # Find which corresponding side is the edge to the other facet
    reciprocal_sides = np.flatnonzero(nabrs[jface] == iface)
    if reciprocal_sides.size != 1:
        raise GraphTopologyError(
            f"Triangles {iface} and {jface} do not have reciprocal neighbours."
        )
    jside = int(reciprocal_sides[0])

    l = int(faces[iface, iside])  # Not-edge vertex
    j = int(faces[iface, (iside + 1) % 3])  # Edge vertex 1
    k = int(faces[iface, (iside + 2) % 3])  # Edge vertex 2
    if not ({j, k} <= set(map(int, faces[jface]))):
        raise GraphTopologyError(
            f"Neighbouring triangle facets {iface} and {jface} do not share one edge."
        )
    m = int(faces[jface, jside])  # Not edge vertex of the other triangle

    if len({l, m, j, k}) != 4:
        raise GraphTopologyError(
            "The incident triangle facets do not form a quadrilateral."
        )

    o_lmj = orient(vtxs[l], vtxs[m], vtxs[j], backend="python")
    o_lmk = orient(vtxs[l], vtxs[m], vtxs[k], backend="python")
    if o_lmj == 0 or o_lmk == 0:
        raise GraphTopologyError("The flipped edge would create a degenerate triangle.")
    if (o_lmj > 0) == (o_lmk > 0):
        raise GraphTopologyError(
            "The incident triangle facets do not form a convex quadrilateral."
        )

    f_faces = np.array(faces, dtype=NpCanonIndex, order="C", copy=True)
    f_faces[iface] = (l, j, m)
    f_faces[jface] = (l, m, k)
    f_nabrs = _update_flipped_neighbours(nabrs, iface, iside, jface, jside)
    return f_faces, f_nabrs


def _find_crossing_edges(
    vtxs: NDArray[NpCoords],
    faces: NDArray[NpCanonIndex],
    nabrs: NDArray[NpCanonIndex],
    edge: tuple[int, int],
) -> list[tuple[int, int, tuple[int, int]]]:
    """
    Finds unique interior edges properly crossing a constraint edge.

    Parameters
    ----------
    vtxs : NDArray[NpCoords]
        Vertex coordinates of shape `(N, 2)`.
    faces : NDArray[NpCanonIndex]
        Triangle vertex index matrix of shape `(F, 3)`.
    nabrs : NDArray[NpCanonIndex]
        Triangle neighbour index matrix of shape `(F, 3)`.
    edge : tuple[int, int]
        Endpoint indices `(u, v)` defining the target constraint
        line segment.

    Returns
    -------
    xngs : list[tuple[int, int, tuple[int, int]]]
        List of crossing mesh edge descriptors. Each entry is a
        tuple containing:
        - `iface`: Facet index owning the edge.
        - `iside`: Local side index (0, 1, or 2) corresponding to
            the edge.
        - `key`: Canonical edge representation `(a, b)` with
            `a < b`.
    """
    # Extract unique interior edges as (triangle, side) pairs,
    # selecting the smaller triangle ID to avoid double counting
    nfaces = faces.shape[0]
    ifaces = np.repeat(np.arange(nfaces), 3)
    isides = np.tile(np.arange(3), nfaces)
    flat_nabrs = nabrs.ravel()
    unq_intr_edges = (flat_nabrs >= 0) & (ifaces < flat_nabrs)
    ifaces = ifaces[unq_intr_edges]
    isides = isides[unq_intr_edges]

    # Gather endpoint indices and coordinates for each candidate mesh edge
    l = faces[ifaces, (isides + 1) % 3]
    m = faces[ifaces, (isides + 2) % 3]
    vtxs_ = np.asarray(vtxs, dtype=np.result_type(vtxs.dtype, np.int64))
    vj: NDArray[NpCoords] = vtxs_[edge[0]]
    vk: NDArray[NpCoords] = vtxs_[edge[1]]
    vl: NDArray[NpCoords] = vtxs_[faces[ifaces, (isides + 1) % 3]]
    vm: NDArray[NpCoords] = vtxs_[faces[ifaces, (isides + 2) % 3]]

    # Compute orientation of mesh edge endpoints relative to constraint vector
    vjk = vk - vj
    vjl = vl - vj
    vjm = vm - vj
    o_jkl: NDArray[NpCoords] = vjk[0] * vjl[:, 1] - vjk[1] * vjl[:, 0]
    o_jkm: NDArray[NpCoords] = vjk[0] * vjm[:, 1] - vjk[1] * vjm[:, 0]

    # Compute orientation of constraint endpoints relative to mesh edge vectors
    vlm = vm - vl
    vlj = vj - vl
    vlk = vk - vl
    o_lmj = vlm[:, 0] * vlj[:, 1] - vlm[:, 1] * vlj[:, 0]
    o_lmk = vlm[:, 0] * vlk[:, 1] - vlm[:, 1] * vlk[:, 0]

    # Classify proper line-segment crossings with strict opposite orientations
    xng: NDArray[np.bool_] = (
        ((o_jkl != 0) & (o_jkm != 0))  # Non-collinear endpoints
        & ((o_lmj != 0) & (o_lmk != 0))  # Non-collinear endpoints
        & ((o_jkl > 0) != (o_jkm > 0))
        & ((o_lmj > 0) != (o_lmk > 0))
    )

    # Pack crossing edge descriptors into output list
    return [
        (int(iface), int(iside), _canonical_edge(int(a), int(b)))
        for iface, iside, a, b in zip(ifaces[xng], isides[xng], l[xng], m[xng])
    ]


def _mesh_contains_edge(faces: NDArray[NpCanonIndex], edge: tuple[int, int]) -> bool:
    """
    Checks whether a mesh contains a specified edge.

    Parameters
    ----------
    faces : NDArray[NpCanonIndex]
        Triangle vertex index matrix of shape (F, 3).
    edge : tuple[int, int]
        Endpoint indices `(u, v)` defining the target edge.

    Returns
    -------
    contains_edge : bool
        `True` if both endpoints co-occur in at least 1 facet,
        `False` otherwise.
    """
    j, k = edge
    return bool(np.any(np.any(faces == j, axis=1) & np.any(faces == k, axis=1)))


def _mesh_topology_key(faces: NDArray[NpCanonIndex]) -> bytes:
    """
    Returns a canonical byte representation of a facet topology.
    """
    canonical = np.sort(faces, axis=1)
    order = np.lexsort((canonical[:, 2], canonical[:, 1], canonical[:, 0]))
    return np.ascontiguousarray(canonical[order]).tobytes()


def recover_constraint_edge(
    vtxs: NDArray[NpCoords],
    faces: NDArray[NpCanonIndex],
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
    Existing locked edges are preserved and never flipped. The input
    arrays are not modified.

    Parameters
    ----------
    vtxs : NDArray[NpCoords]
        Vertex coordinate matrix of shape (N, 2).
    faces : NDArray[NpCanonIndex]
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
        computed from `faces`.
        Default input is `None`.

    Returns
    -------
    r_faces : NDArray[NpCanonIndex]
        Updated triangle vertex index matrix of shape (F, 3).
    nabrs : NDArray[NpCanonIndex]
        Updated triangle neighbour matrix of shape (F, 3).

    Raises
    ------
    ValueError
        If `nabrs` is supplied but its shape does not match
        `faces`.
    GraphTopologyError
        If constraint edge crosses a locked edge, no flippable edge
        crosses the constraint, or edge flips fail to make progress
        towards edge recovery.
    """
    u, v = map(int, edge)

    target = _canonical_edge(u, v)
    locked = {_canonical_edge(*locked_edge) for locked_edge in (locked_edges or set())}

    r_faces = np.array(faces, dtype=NpCanonIndex, order="C", copy=True)
    if nabrs is None:
        nabrs, _ = find_facet_neighbours(r_faces)
    else:
        nabrs = np.asarray(nabrs)
        if nabrs.shape != r_faces.shape:
            raise ValueError(
                "Neighbours must have the same shape as triangle facets, "
                + f"but got {nabrs.shape} and {r_faces.shape}."
            )
    if _mesh_contains_edge(r_faces, target):
        return r_faces, nabrs

    visited = {_mesh_topology_key(r_faces)}
    target_recovered = False

    while not target_recovered:
        xngs = _find_crossing_edges(vtxs, r_faces, nabrs, target)
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
        for iface, iside, _ in xngs:
            try:
                f_faces, f_nabrs = flip_quadrilateral_edge(
                    vtxs, r_faces, nabrs, iface, iside
                )
            except GraphTopologyError:
                continue

            f_xngs = _find_crossing_edges(vtxs, f_faces, f_nabrs, target)
            f_has_target = _mesh_contains_edge(f_faces, target)
            f_key = _mesh_topology_key(f_faces)
            if f_key in visited:
                continue
            if not f_has_target and len(f_xngs) > len(xngs):
                continue

            r_faces = f_faces
            nabrs = f_nabrs
            target_recovered = f_has_target
            visited.add(f_key)
            progressed = True
            break

        if not progressed:
            raise GraphTopologyError(
                f"No legal edge flip makes progress recovering constraint {target}."
            )

    return r_faces, nabrs


def recover_constraint_edges(
    vtxs: NDArray[NpCoords],
    faces: NDArray[NpCanonIndex],
    edges: NDArray[NpCanonIndex],
) -> tuple[NDArray[NpCanonIndex], NDArray[NpCanonIndex]]:
    """
    Recovers a set of non-crossing constraint edges in a
    triangulation.

    Constraints are recovered sequentially using edge flips. Each
    successfully recovered edge is added to a locked set so
    subsequent recovery steps do not alter or remove it. The input
    arrays are not modified.

    Parameters
    ----------
    vtxs : NDArray[NpCoords]
        Vertex coordinate matrix of shape (N, 2).
    faces : NDArray[NpCanonIndex]
        Triangle vertex index matrix of shape (F, 3).
    edges : NDArray[NpCanonIndex]
        Constraint edge matrix of shape (E, 2) containing vertex
        index pairs.

    Returns
    -------
    r_faces : NDArray[NpCanonIndex]
        Updated triangle vertex index matrix of shape (F, 3).
    nabrs : NDArray[NpCanonIndex]
        Updated triangle neighbour matrix of shape (F, 3).

    Raises
    ------
    GraphTopologyError
        If any constraint edge fails to recover or is absent from
        the final mesh topology.
    """
    r_faces = np.array(faces, dtype=NpCanonIndex, order="C", copy=True)
    nabrs, edge_owners = find_facet_neighbours(r_faces)
    initial_mesh_edges = set(edge_owners)
    locked: set[tuple[int, int]] = set()

    for iedge, edge in enumerate(edges):
        target = _canonical_edge(int(edge[0]), int(edge[1]))
        if target in initial_mesh_edges or target in locked:
            locked.add(target)
            continue
        try:
            r_faces, nabrs = recover_constraint_edge(
                vtxs,
                r_faces,
                target,
                locked_edges=locked,
                nabrs=nabrs,
            )
        except (GraphTopologyError, IndexError, ValueError) as exc:
            raise type(exc)(
                f"Failed to recover constraint edge {iedge} {target}: {exc}"
            ) from exc
        locked.add(target)

    _, edge_owners = find_facet_neighbours(r_faces)
    missing = [edge for edge in locked if edge not in edge_owners]
    if missing:
        raise GraphTopologyError(
            f"Recovered constraint edge {missing[0]} is absent from the final mesh."
        )

    return r_faces, nabrs
