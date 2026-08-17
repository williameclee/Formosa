"""
Provides the public API for Delaunay triangulation and constraint
recovery.

This module dispatches to the Python or Fortran backend and
normalises native inputs, outputs, and errors.

Created: 2026-08-12, En-Chi Lee (williameclee@gmail.com)
Last modified: 2026-08-17, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np

from formosa.geomorphology._native import meshing_triangulation as tri_f
import formosa.geomorphology.meshing._backends.triangulation_py as tri_py
from formosa.geomorphology.drainage.network import GraphTopologyError

from typing import Optional
from numpy.typing import NDArray
from formosa.utils import Backend, raise_fortran_error
from formosa.utils.typing import NpCoords, NpCanonIndex

_TRIANGULATION_ERRORS = {
    1: (ValueError, "invalid triangulation input"),
    2: (MemoryError, "unable to allocate triangulation workspace"),
    3: (RuntimeError, "triangulation capacity exceeded"),
    4: (GraphTopologyError, "point set did not produce a valid triangulation"),
}


def _to_fortran_coords(vtxs: NDArray[NpCoords]) -> NDArray[np.int32]:
    """Converts ``(V, 2)`` coordinates to native ``(2, V)`` layout."""
    return np.asfortranarray(vtxs.T, dtype=np.int32)


def _to_fortran_indices(indices: NDArray[NpCanonIndex]) -> NDArray[np.int32]:
    """Converts zero-based row-major indices to one-based native layout."""
    return np.asfortranarray(indices.T, dtype=np.int32) + 1


def _to_fortran_neighbours(nabrs: NDArray[NpCanonIndex]) -> NDArray[np.int32]:
    """Converts neighbours to native layout without changing ``-1`` sentinels."""
    nabrs_f = np.array(nabrs.T, dtype=np.int32, order="F", copy=True)
    nabrs_f[nabrs_f >= 0] += 1
    return nabrs_f


def _from_fortran_indices(indices_f: NDArray[np.int32]) -> NDArray[NpCanonIndex]:
    """Converts one-based native indices to zero-based row-major layout."""
    return np.ascontiguousarray(indices_f.T, dtype=NpCanonIndex) - 1


def _from_fortran_neighbours(nabrs_f: NDArray[np.int32]) -> NDArray[NpCanonIndex]:
    """Converts native neighbours to row-major layout, preserving ``-1``."""
    nabrs = np.ascontiguousarray(nabrs_f.T, dtype=NpCanonIndex)
    nabrs[nabrs >= 0] -= 1
    return nabrs


def _validate_fortran_coords(vtxs: NDArray[NpCoords]) -> None:
    """Validates coordinates against the native integer representation."""
    if not np.issubdtype(vtxs.dtype, np.integer):
        raise TypeError(
            "The Fortran triangulation backend requires integer coordinates, "
            + f"but got {vtxs.dtype}."
        )
    int32_info = np.iinfo(np.int32)
    if np.any(vtxs < int32_info.min) or np.any(vtxs > int32_info.max):
        raise OverflowError(
            "The Fortran triangulation backend requires coordinates "
            + "representable as int32."
        )


def _validate_fortran_vertex_ids(faces: NDArray[NpCanonIndex]) -> None:
    """Validates zero-based vertex IDs before conversion to one-based int32."""
    if np.any(faces >= np.iinfo(np.int32).max):
        raise OverflowError(
            "The Fortran triangulation backend requires vertex IDs "
            + "smaller than the int32 maximum."
        )


def _validate_triangulate_points(vtxs: NDArray[NpCoords]) -> None:
    if vtxs.ndim != 2 or vtxs.shape[1] != 2:
        raise ValueError("Vertices must have shape (V, 2), " + f"but got {vtxs.shape}.")
    if vtxs.shape[0] < 3:
        raise ValueError(
            "At least 3 vertices are required, " + f"but only got {vtxs.shape[0]}."
        )
    n_unq_pts = np.unique(vtxs, axis=0).shape[0]
    if n_unq_pts != vtxs.shape[0]:
        raise ValueError(
            "Vertices must be unique, "
            + f"but found {vtxs.shape[0]-n_unq_pts} duplicates."
        )


def _canonicalise_facets(
    faces: NDArray[NpCanonIndex],
) -> NDArray[NpCanonIndex]:
    """
    Returns CCW triangles in a deterministic vertex and row order.
    """
    faces = np.asarray(faces, dtype=NpCanonIndex, order="C")

    # Cyclic rotation such that the smallest index appears first
    starts = np.argmin(faces, axis=1)
    offsets = np.arange(3)
    faces = np.take_along_axis(faces, (starts[:, np.newaxis] + offsets) % 3, axis=1)

    # Sort by first, then second, then third index
    order = np.lexsort((faces[:, 2], faces[:, 1], faces[:, 0]))
    return np.ascontiguousarray(faces[order], dtype=NpCanonIndex)


def _canonicalise_facet_topology(
    faces: NDArray[NpCanonIndex],
    nabrs: NDArray[NpCanonIndex],
) -> tuple[NDArray[NpCanonIndex], NDArray[NpCanonIndex]]:
    """
    Returns facets and their neighbours in a canonical order.

    Neighbour columns follow facet vertex rotations, neighbour rows
    follow facet sorting, and neighbour IDs are remapped to the new
    facet IDs.
    """
    faces = np.asarray(faces, dtype=NpCanonIndex, order="C")
    nabrs = np.asarray(nabrs, dtype=NpCanonIndex, order="C")

    starts = np.argmin(faces, axis=1)
    offsets = np.arange(3)
    rotations = (starts[:, np.newaxis] + offsets) % 3
    faces = np.take_along_axis(faces, rotations, axis=1)
    nabrs = np.take_along_axis(nabrs, rotations, axis=1)

    order = np.lexsort((faces[:, 2], faces[:, 1], faces[:, 0]))
    faces = faces[order]
    nabrs = nabrs[order]

    old_to_new = np.empty(faces.shape[0], dtype=NpCanonIndex)
    old_to_new[order] = np.arange(faces.shape[0], dtype=NpCanonIndex)
    interior = nabrs >= 0
    nabrs[interior] = old_to_new[nabrs[interior]]

    return (
        np.ascontiguousarray(faces, dtype=NpCanonIndex),
        np.ascontiguousarray(nabrs, dtype=NpCanonIndex),
    )


def triangulate_points(
    vtxs: NDArray[NpCoords], backend: Backend = "fortran"
) -> NDArray[NpCanonIndex]:
    """
    Computes an unconstrained Delaunay triangulation of 2D points.

    Each counterclockwise triangle starts with its smallest vertex
    ID, and triangles are ordered lexicographically.

    Parameters
    ----------
    vtxs : NDArray[number], shape (V, 2)
        Unique vertex coordinates. At least three non-collinear
        points are required.
    backend : {"fortran", "python"}, optional
        Computational backend.
        Default backend is `"fortran"`.

    Returns
    -------
    faces : NDArray[int32], shape (F, 3)
        Counterclockwise triangle vertex IDs in canonical,
        lexicographic order.

    Raises
    ------
    ValueError
        If the vertices have an invalid shape, contain duplicates,
        are too few to triangulate, or the backend is unsupported.
    TypeError
        If the coordinates are not numeric, or the Fortran backend
        receives non-integer coordinates.
    OverflowError
        If the Fortran backend receives coordinates outside the
        `int32` range.
    GraphTopologyError
        If the points are collinear or do not produce a valid
        triangulation.
    MemoryError
        If the Fortran backend cannot allocate its workspace.
    RuntimeError
        If the Fortran triangulation capacity is exceeded.

    Notes
    -----
    The native (Fortran) backend currently accepts only coordinates
    representable as `int32`.
    """
    vtxs = np.asarray(vtxs)
    _validate_triangulate_points(vtxs)

    match backend:
        case "python":
            faces = tri_py.triangulate_points(vtxs)
        case "fortran":
            _validate_fortran_coords(vtxs)
            vtxs_f = _to_fortran_coords(vtxs)
            faces_f, nfaces, err_code = tri_f.triangulate_points(vtxs_f)
            raise_fortran_error(
                "triangulate_points",
                err_code,
                errors=_TRIANGULATION_ERRORS,
            )
            faces = _from_fortran_indices(faces_f[:, :nfaces])
        case _:
            raise ValueError(f"Unknown backend: {backend}")
    return _canonicalise_facets(faces)


def find_facet_neighbours(
    faces: NDArray[NpCanonIndex], backend: Backend = "fortran"
) -> NDArray[NpCanonIndex]:
    """
    Finds the triangle adjacent across each triangle side.

    Side `i` lies opposite vertex `i`. Boundary sides have
    neighbour ID `-1`.

    Parameters
    ----------
    faces : NDArray[int], shape (F, 3)
        Non-negative, 0-based triangle vertex IDs.
    backend : {"fortran", "python"}, optional
        Computational backend.
        Default backend is `"fortran"`.

    Returns
    -------
    nabrs : NDArray[int32], shape (F, 3)
        0-based neighbouring triangle IDs, with `-1` at the mesh
        boundary.

    Raises
    ------
    ValueError
        If the triangle array has an invalid shape, contains negative
        vertex IDs, or the backend is unsupported.
    TypeError
        If the triangle vertex IDs are not integers.
    OverflowError
        If the Fortran backend receives a vertex ID that cannot be
        converted safely to its native representation.
    GraphTopologyError
        If the triangles do not form a valid manifold mesh.
    MemoryError
        If the Fortran backend cannot allocate its workspace.
    """
    faces = np.asarray(faces)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Triangles must have shape (F, 3).")
    if not np.issubdtype(faces.dtype, np.integer):
        raise TypeError("Triangle vertex IDs must be integers.")
    if np.any(faces < 0):
        raise ValueError("Triangle vertex IDs must be non-negative.")

    match backend:
        case "python":
            nabrs, _ = tri_py.find_facet_neighbours(faces)
        case "fortran":
            _validate_fortran_vertex_ids(faces)
            faces_f = _to_fortran_indices(faces)
            nabrs_f, err_code = tri_f.find_facet_neighbours(faces_f)
            raise_fortran_error(
                "find_facet_neighbours", err_code, errors=_TRIANGULATION_ERRORS
            )
            nabrs = _from_fortran_neighbours(nabrs_f)
        case _:
            raise ValueError(f"Unknown backend: {backend}")

    return np.ascontiguousarray(nabrs, dtype=NpCanonIndex)


def flip_quadrilateral_edge(
    vtxs: NDArray[NpCoords],
    faces: NDArray[NpCanonIndex],
    iface: int,
    iside: int,
    nabrs: Optional[NDArray[NpCanonIndex]] = None,
    backend: Backend = "fortran",
) -> tuple[NDArray[NpCanonIndex], NDArray[NpCanonIndex]]:
    """
    Flips an interior triangle edge in a convex quadrilateral.

    The selected side is opposite vertex `iside` of triangle
    `iface`. Input arrays are not modified. If `nabrs` is omitted,
    neighbours are computed with the selected backend before the
    flip.

    Parameters
    ----------
    vtxs : NDArray[number], shape (V, 2)
        Vertex coordinates.
    faces : NDArray[int], shape (F, 3)
        Counterclockwise, 0-based triangle vertex IDs.
    iface : int
        ID of the triangle containing the edge to flip.
    iside : int
        Local side ID in the range `[0, 3)`.
    nabrs : NDArray[int], shape (F, 3), optional
        Triangle neighbours, with `-1` at the mesh boundary. They are
        computed when omitted.
        Default input is `None`.
    backend : {"fortran", "python"}, optional
        Computational backend.
        Default backend is `"fortran"`.

    Returns
    -------
    f_faces : NDArray[int32], shape (F, 3)
        Triangle vertex IDs after replacing the selected diagonal.
    f_nabrs : NDArray[int32], shape (F, 3)
        Triangle neighbours after the flip.

    Raises
    ------
    ValueError
        If an input array has an invalid shape or the backend is
        unsupported.
    TypeError
        If the Fortran backend receives non-integer coordinates.
    IndexError
        If a vertex, neighbour, triangle, or side ID is out of bounds.
    OverflowError
        If the Fortran backend cannot represent an input coordinate or
        vertex ID.
    GraphTopologyError
        If the selected edge is on the mesh boundary or its adjacent
        triangles do not form a flippable convex quadrilateral.
    """
    vtxs = np.asarray(vtxs)
    faces = np.asarray(faces)
    if nabrs is not None:
        nabrs = np.asarray(nabrs)
    else:
        nabrs = find_facet_neighbours(faces, backend=backend)

    if vtxs.ndim != 2 or vtxs.shape[1] != 2:
        raise ValueError(f"Vertices must have shape (V, 2), but got {vtxs.shape}.")

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Triangles must have shape (F, 3), but got {faces.shape}.")
    elif np.any(faces < 0) or np.any(faces >= vtxs.shape[0]):
        raise IndexError("Some triangles reference invalid vertex.")

    if nabrs.shape != faces.shape:
        raise ValueError(
            "Neighbours must have the same shape as triangles, "
            + f"but got {nabrs.shape} and {faces.shape}."
        )
    elif np.any(nabrs < -1) or np.any(nabrs >= faces.shape[0]):
        raise IndexError("Some triangle sides reference invalid neighbour.")

    if iface < 0 or iface >= faces.shape[0]:
        raise IndexError(f"Triangle ID {iface} is out of bounds.")
    if iside < 0 or iside >= 3:
        raise IndexError(f"Triangle side ID {iside} is out of bounds.")

    match backend:
        case "python":
            f_faces, f_nabrs = tri_py.flip_quadrilateral_edge(
                vtxs, faces, nabrs, iface, iside
            )
        case "fortran":
            _validate_fortran_coords(vtxs)
            _validate_fortran_vertex_ids(faces)
            vtxs_f = _to_fortran_coords(vtxs)
            faces_f = _to_fortran_indices(faces)
            nabrs_f = _to_fortran_neighbours(nabrs)
            _, err_code = tri_f.flip_quadrilateral_edge(
                vtxs_f, faces_f, nabrs_f, iface + 1, iside + 1
            )
            raise_fortran_error(
                "flip_quadrilateral_edge", err_code, errors=_TRIANGULATION_ERRORS
            )
            f_faces = _from_fortran_indices(faces_f)
            f_nabrs = _from_fortran_neighbours(nabrs_f)
        case _:
            raise ValueError(f"Unknown backend: {backend}")
    return f_faces, f_nabrs


def _find_crossing_edges(
    vtxs: NDArray[NpCoords],
    faces: NDArray[NpCanonIndex],
    nabrs: NDArray[NpCanonIndex],
    edge: tuple[int, int],
    backend: Backend,
) -> list[tuple[int, int, tuple[int, int]]]:
    """
    Dispatches the internal proper-crossing edge query to a backend.

    Notes
    -----
    Function not meant for public use.
    """
    match backend:
        case "python":
            return tri_py._find_crossing_edges(vtxs, faces, nabrs, edge)
        case "fortran":
            vtxs_f = _to_fortran_coords(vtxs)
            faces_f = _to_fortran_indices(faces)
            nabrs_f = _to_fortran_neighbours(nabrs)
            edge_f = _to_fortran_indices(np.asarray(edge, dtype=NpCanonIndex))
            xngs_f, nxngs, err_code = tri_f.find_crossing_edges(
                vtxs_f, faces_f, nabrs_f, edge_f
            )
            raise_fortran_error(
                "find_crossing_edges", err_code, errors=_TRIANGULATION_ERRORS
            )
            xngs = _from_fortran_indices(xngs_f[:, :nxngs])
            return [
                (int(iface), int(iside), (int(j), int(k)))
                for iface, iside, j, k in xngs
            ]
        case _:
            raise ValueError(f"Unknown backend: {backend}")


def _validate_constraint_mesh(
    vtxs: NDArray[NpCoords], faces: NDArray[NpCanonIndex]
) -> None:
    """
    Validates mesh inputs shared by constraint-recovery APIs.
    """
    if vtxs.ndim != 2 or vtxs.shape[1] != 2:
        raise ValueError(f"Vertices must have shape (V, 2), but got {vtxs.shape}.")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(
            f"Triangle facets must have shape (F, 3), but got {faces.shape}."
        )
    if not np.issubdtype(faces.dtype, np.integer):
        raise TypeError("Triangle facet vertex IDs must be integers.")
    if np.any(faces < 0) or np.any(faces >= vtxs.shape[0]):
        raise IndexError("Some triangle facets reference invalid vertex.")


def _validate_constraint_edges(edges: NDArray[NpCanonIndex], nvtxs: int) -> None:
    """
    Validates a constraint-edge matrix.
    """
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(
            "Constraint edges must have shape (E, 2), " + f"but got {edges.shape}."
        )
    if not np.issubdtype(edges.dtype, np.integer):
        raise TypeError(
            "Constraint edge vertex IDs must be integers, " + f"but got {edges.dtype}."
        )
    if np.any(edges < 0):
        raise IndexError("Constraint edge vertex IDs must be non-negative.")
    if np.any(edges >= nvtxs):
        raise IndexError("Constraint edge vertex IDs are out of bounds.")
    if (count := int(np.sum(edges[:, 0] == edges[:, 1]))) > 0:
        raise ValueError(
            "Constraint edges cannot be self-edges, " + f"but found {count} self-edges."
        )


def recover_constraint_edge(
    vtxs: NDArray[NpCoords],
    faces: NDArray[NpCanonIndex],
    edge: tuple[int, int],
    locked_edges: set[tuple[int, int]] | None = None,
    nabrs: Optional[NDArray[NpCanonIndex]] = None,
    backend: Backend = "fortran",
) -> tuple[NDArray[NpCanonIndex], NDArray[NpCanonIndex]]:
    """
    Recovers a constraint edge from a Delaunay triangulation using
    iterative edge flips.

    Input arrays are not modified. Existing mesh edges listed in
    `locked_edges` are preserved throughout recovery.

    Parameters
    ----------
    vtxs : NDArray[number], shape (V, 2)
        Vertex coordinates.
    faces : NDArray[int], shape (F, 3)
        Counterclockwise, 0-based triangle vertex IDs.
    edge : tuple[int, int]
        Vertex IDs of the constraint edge to recover.
    locked_edges : set[tuple[int, int]], optional
        Existing mesh edges that must not be flipped.
        Default input is `None`.
    nabrs : NDArray[int], shape (F, 3), optional
        Triangle neighbours, with `-1` at the mesh boundary.
        They are computed when omitted.
        Default input is `None`.
    backend : {"fortran", "python"}, optional
        Computational backend.
        Default backend is `"fortran"`.

    Returns
    -------
    r_faces : NDArray[int32], shape (F, 3)
        Triangle vertex IDs for a mesh containing the constraint.
    r_nabrs : NDArray[int32], shape (F, 3)
        Triangle neighbours for the recovered mesh.

    Raises
    ------
    ValueError
        If an input has an invalid shape, an edge is a self-edge, or
        the backend is unsupported.
    TypeError
        If an edge or neighbour ID is not an integer, or the Fortran
        backend receives non-integer coordinates.
    IndexError
        If a vertex or neighbour ID is out of bounds.
    OverflowError
        If the Fortran backend cannot represent an input coordinate
        or vertex ID.
    GraphTopologyError
        If no legal sequence of edge flips can recover the
        constraint while preserving the locked edges.
    """
    vtxs = np.asarray(vtxs)
    faces = np.asarray(faces)
    _validate_constraint_mesh(vtxs, faces)

    edge_array = np.asarray(edge)
    if edge_array.shape != (2,):
        raise ValueError(
            "Constraint edge must have shape (2,), " + f"but got {edge_array.shape}."
        )
    if not np.issubdtype(edge_array.dtype, np.integer):
        raise TypeError("Constraint edge vertex IDs must be integers.")
    if np.any(edge_array < 0) or np.any(edge_array >= vtxs.shape[0]):
        raise IndexError("Constraint edge vertex IDs are out of bounds.")
    if edge_array[0] == edge_array[1]:
        raise ValueError("Constraint edge cannot be a self-edge.")

    j, k = map(int, edge_array)
    target = tri_py._canonical_edge(j, k)
    locked: set[tuple[int, int]] = set()
    for locked_edge in locked_edges or set():
        locked_array = np.asarray(locked_edge)
        if locked_array.shape != (2,):
            raise ValueError("Locked edges must contain vertex pairs.")
        if not np.issubdtype(locked_array.dtype, np.integer):
            raise TypeError("Locked edge vertex IDs must be integers.")
        l, m = map(int, locked_array)
        if l < 0 or l >= vtxs.shape[0] or m < 0 or m >= vtxs.shape[0]:
            raise IndexError("Locked edge vertex IDs are out of bounds.")
        if l == m:
            raise ValueError("Locked edges cannot be self-edges.")
        locked.add(tri_py._canonical_edge(l, m))

    if nabrs is None:
        nabrs = find_facet_neighbours(faces, backend=backend)
    else:
        nabrs = np.asarray(nabrs)
        if nabrs.shape != faces.shape:
            raise ValueError(
                "Neighbours must have the same shape as triangles, "
                + f"but got {nabrs.shape} and {faces.shape}."
            )
        if not np.issubdtype(nabrs.dtype, np.integer):
            raise TypeError("Neighbour triangle IDs must be integers.")
        if np.any(nabrs < -1) or np.any(nabrs >= faces.shape[0]):
            raise IndexError("Some triangle sides reference invalid neighbour.")

    match backend:
        case "python":
            r_faces, r_nabrs = tri_py.recover_constraint_edge(
                vtxs,
                faces,
                target,
                locked_edges=locked,
                nabrs=nabrs,
            )
        case "fortran":
            _validate_fortran_coords(vtxs)
            _validate_fortran_vertex_ids(faces)

            vtxs_f = _to_fortran_coords(vtxs)
            faces_f = _to_fortran_indices(faces)
            nabrs_f = _to_fortran_neighbours(nabrs)
            edge_f = _to_fortran_indices(np.asarray(target, dtype=NpCanonIndex))
            if locked:
                locked_f = _to_fortran_indices(
                    np.asarray(sorted(locked), dtype=NpCanonIndex)
                )
            else:
                locked_f = np.empty((2, 0), dtype=np.int32, order="F")

            err_code = tri_f.recover_constraint_edge(
                vtxs_f, faces_f, nabrs_f, edge_f, locked_f
            )
            raise_fortran_error(
                "recover_constraint_edge",
                err_code,
                errors=_TRIANGULATION_ERRORS,
            )
            r_faces = _from_fortran_indices(faces_f)
            r_nabrs = _from_fortran_neighbours(nabrs_f)
        case _:
            raise ValueError(f"Unknown backend: {backend}")

    return (
        np.ascontiguousarray(r_faces, dtype=NpCanonIndex),
        np.ascontiguousarray(r_nabrs, dtype=NpCanonIndex),
    )


def recover_constraint_edges(
    vtxs: NDArray[NpCoords],
    faces: NDArray[NpCanonIndex],
    edges: NDArray[NpCanonIndex],
    backend: Backend = "fortran",
) -> tuple[NDArray[NpCanonIndex], NDArray[NpCanonIndex]]:
    """
    Recovers constraint edges than may not be present in the input
    Delaunay triangulation sequentially using edge flips.

    Every recovered constraint is locked before the next is
    processed. Input arrays are not modified. Returned facets start
    with their smallest vertex ID and are ordered lexicographically;
    neighbour rows, columns, and IDs follow that canonical order.

    Parameters
    ----------
    vtxs : NDArray[number], shape (V, 2)
        Vertex coordinates.
    faces : NDArray[int], shape (F, 3)
        Counterclockwise, 0-based triangle vertex IDs.
    edges : NDArray[int], shape (E, 2)
        Constraint vertex pairs, processed in row order.
    backend : {"fortran", "python"}, optional
        Computational backend.
        Default backend is `"fortran"`.

    Returns
    -------
    r_faces : NDArray[int32], shape (F, 3)
        Triangle vertex IDs for a mesh containing every constraint,
        in canonical lexicographic order.
    nabrs : NDArray[int32], shape (F, 3)
        Triangle neighbours for the recovered mesh.

    Raises
    ------
    ValueError
        If an input array has an invalid shape, a constraint is a
        self-edge, or the backend is unsupported.
    TypeError
        If triangle or constraint vertex IDs are not integers, or
        the Fortran backend receives non-integer coordinates.
    IndexError
        If a triangle or constraint references an invalid vertex.
    OverflowError
        If the Fortran backend cannot represent an input coordinate
        or vertex ID.
    GraphTopologyError
        If a constraint cannot be recovered without changing an
        earlier constraint.
    MemoryError
        If the Fortran backend cannot allocate its workspace.
    RuntimeError
        If the Fortran backend exceeds its recovery capacity.
    """
    vtxs = np.asarray(vtxs)
    faces = np.asarray(faces)
    edges = np.asarray(edges)
    _validate_constraint_mesh(vtxs, faces)
    _validate_constraint_edges(edges, vtxs.shape[0])

    match backend:
        case "python":
            r_faces, r_nabrs = tri_py.recover_constraint_edges(vtxs, faces, edges)
        case "fortran":
            _validate_fortran_coords(vtxs)
            _validate_fortran_vertex_ids(faces)
            vtxs_f = _to_fortran_coords(vtxs)
            faces_f = _to_fortran_indices(faces)
            edges_f = _to_fortran_indices(edges)

            r_nabrs_f, failed_edge, err_code = tri_f.recover_constraint_edges(
                vtxs_f, faces_f, edges_f
            )
            if err_code != 0 and failed_edge > 0:
                failed_index = int(failed_edge) - 1
                u, v = map(int, edges[failed_index])
                target = tri_py._canonical_edge(u, v)
                try:
                    raise_fortran_error(
                        "recover_constraint_edges",
                        err_code,
                        errors=_TRIANGULATION_ERRORS,
                    )
                except (
                    GraphTopologyError,
                    ValueError,
                    MemoryError,
                    RuntimeError,
                ) as exc:
                    raise type(exc)(
                        f"Failed to recover constraint edge {failed_index} "
                        + f"{target}: {exc}"
                    ) from exc
            raise_fortran_error(
                "recover_constraint_edges",
                err_code,
                errors=_TRIANGULATION_ERRORS,
            )
            r_faces = _from_fortran_indices(faces_f)
            r_nabrs = _from_fortran_neighbours(r_nabrs_f)
        case _:
            raise ValueError(f"Unknown backend: {backend}")

    return _canonicalise_facet_topology(r_faces, r_nabrs)
