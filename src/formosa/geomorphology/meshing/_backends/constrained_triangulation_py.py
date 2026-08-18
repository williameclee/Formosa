"""
Recovers constrained edges using the Python backend.

This internal module adds constraints to an existing unconstrained
triangulation through edge flipping. Base triangulation and
facet-neighbour construction are implemented separately in
`triangulation_py.py`.

Created: 2026-08-17, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np

from formosa.geomorphology.geometry import orient
from formosa.geomorphology.drainage.network import GraphTopologyError
from formosa.geomorphology.meshing._backends.triangulation_py import (
    _canonical_edge,
    find_facet_neighbours,
)

from numpy.typing import NDArray
from formosa.utils.typing import NpCoords, NpCanonIndex


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
