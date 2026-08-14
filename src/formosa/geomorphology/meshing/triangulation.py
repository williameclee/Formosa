"""
Provides the public API for unconstrained Delaunay triangulation.

This module dispatches to the Python or FORTRAN backend and
normalises native inputs, outputs, and errors.

Created: 2026-08-12, En-Chi Lee (williameclee@gmail.com)
Last modified: 2026-08-14, En-Chi Lee (williameclee@gmail.com)
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


def _canonicalise_triangles(
    triangles: NDArray[NpCanonIndex],
) -> NDArray[NpCanonIndex]:
    """
    Returns CCW triangles in a deterministic vertex and row order.
    """
    triangles = np.asarray(triangles, dtype=NpCanonIndex, order="C")

    # Cyclic rotation such that the smallest index appears first
    starts = np.argmin(triangles, axis=1)
    offsets = np.arange(3)
    triangles = np.take_along_axis(
        triangles, (starts[:, np.newaxis] + offsets) % 3, axis=1
    )

    # Sort by first, then second, then third index
    order = np.lexsort((triangles[:, 2], triangles[:, 1], triangles[:, 0]))
    return np.ascontiguousarray(triangles[order], dtype=NpCanonIndex)


def triangulate_points(
    vtxs: NDArray[NpCoords], backend: Backend = "fortran"
) -> NDArray[NpCanonIndex]:
    """
    Computes an unconstrained Delaunay triangulation of 2D points.

    Each counterclockwise triangle starts with its smallest vertex
    ID, and triangles are ordered lexicographically.

    Parameters
    ----------
    vtxs : NDArray[int]
        Unique vertex coordinate indices.
        At least 3 points are required, and all indices must be
        non-negative.
    backend : {'fortran', 'python'}, optional
        Backend to use for computation.
        `'fortran'` uses the FORTRAN extension for performance,
        while `'python'` uses a pure Python implementation.
        Default backend is `'fortran'`.

    Returns
    -------
    NDArray[int32], shape (F, 3)
        Counterclockwise triangle vertex IDs.

    Notes
    -----
    The native (FORTRAN) backend currently accepts only coordinates
    representable as `int32`.
    """
    vtxs = np.asarray(vtxs)
    _validate_triangulate_points(vtxs)

    match backend:
        case "python":
            triangles = tri_py.triangulate_points(vtxs)
        case "fortran":
            if not np.issubdtype(vtxs.dtype, np.integer):
                raise TypeError(
                    "The FORTRAN triangulation backend requires integer coordinates, "
                    + f"but got {vtxs.dtype}."
                )
            int32_info = np.iinfo(np.int32)
            if np.any(vtxs < int32_info.min) or np.any(vtxs > int32_info.max):
                raise OverflowError(
                    "The FORTRAN triangulation backend requires coordinates representable as int32, "
                    + f"but detected overflowed coordinates."
                )
            points_f = np.asfortranarray(
                vtxs.T, dtype=np.int32
            )  # No need to convert to 1-based index
            triangles_f, ntris, err_code = tri_f.triangulate_points(points_f)
            raise_fortran_error(
                "triangulate_points",
                err_code,
                errors=_TRIANGULATION_ERRORS,
            )
            triangles = (
                triangles_f[:, :ntris].T.astype(NpCanonIndex, order="C") - 1
            )  # Truncate and convert back to 0-based index
        case _:
            raise ValueError(f"Unknown backend: {backend}")
    return _canonicalise_triangles(triangles)


def find_triangle_neighbours(
    triangles: NDArray[NpCanonIndex], backend: Backend = "fortran"
) -> NDArray[NpCanonIndex]:
    """
    Finds the triangle adjacent across each triangle side.

    Side `i` lies opposite vertex `i`. Boundary sides have
    neighbour ID `-1`.

    Parameters
    ----------
    triangles : NDArray[int], shape (F, 3)
        0-based triangle vertex IDs.
    backend : {'fortran', 'python'}, optional
        Backend to use for computation.
        `'fortran'` uses the FORTRAN extension for performance,
        while `'python'` uses a pure Python implementation.
        Default backend is `'fortran'`.

    Returns
    -------
    NDArray[int32], shape (F, 3)
        0-based neighbouring triangle IDs, with `-1` at the
        mesh boundary.
    """
    triangles = np.asarray(triangles)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("Triangles must have shape (F, 3).")
    if not np.issubdtype(triangles.dtype, np.integer):
        raise TypeError("Triangle vertex IDs must be integers.")
    if np.any(triangles < 0):
        raise ValueError("Triangle vertex IDs must be non-negative.")

    match backend:
        case "python":
            neighbours, _ = tri_py.find_triangle_neighbours(triangles)
        case "fortran":
            int32_info = np.iinfo(np.int32)
            if np.any(triangles >= int32_info.max):
                raise OverflowError(
                    "The FORTRAN triangulation backend requires vertex IDs "
                    + "smaller than the int32 maximum."
                )
            triangles_f = np.asfortranarray(triangles.T, dtype=np.int32) + 1
            neighbours_f, err_code = tri_f.find_triangle_neighbours(triangles_f)
            raise_fortran_error(
                "find_triangle_neighbours",
                err_code,
                errors=_TRIANGULATION_ERRORS,
            )
            neighbours = neighbours_f.T.astype(NpCanonIndex, order="C")
            neighbours[neighbours >= 0] -= 1
        case _:
            raise ValueError(f"Unknown backend: {backend}")

    return np.ascontiguousarray(neighbours, dtype=NpCanonIndex)


def flip_triangle_edge(
    vtxs: NDArray[NpCoords],
    triangles: NDArray[NpCanonIndex],
    itri: int,
    iside: int,
    nabrs: Optional[NDArray[NpCanonIndex]] = None,
    backend: Backend = "python",
) -> tuple[NDArray[NpCanonIndex], NDArray[NpCanonIndex]]:
    vtxs = np.asarray(vtxs)
    triangles = np.asarray(triangles)
    if nabrs is not None:
        nabrs = np.asarray(nabrs)
    else:
        nabrs = find_triangle_neighbours(triangles, backend=backend)

    if vtxs.ndim != 2 or vtxs.shape[1] != 2:
        raise ValueError(f"Vertices must have shape (V, 2), but got {vtxs.shape}.")

    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(
            f"Triangles must have shape (F, 3), but got {triangles.shape}."
        )
    elif np.any(triangles < 0) or np.any(triangles >= vtxs.shape[0]):
        raise IndexError("Some triangles reference invalid vertex.")

    if nabrs.shape != triangles.shape:
        raise ValueError(
            "Neighbours must have the same shape as triangles, "
            + f"but got {nabrs.shape} and {triangles.shape}."
        )
    elif np.any(nabrs < -1) or np.any(nabrs >= triangles.shape[0]):
        raise IndexError("Some triangle sides reference invalid neighbour.")

    if itri < 0 or itri >= triangles.shape[0]:
        raise IndexError(f"Triangle ID {itri} is out of bounds.")
    if iside < 0 or iside >= 3:
        raise IndexError(f"Triangle side ID {iside} is out of bounds.")

    match backend:
        case "python":
            f_triangles, f_nabrs = tri_py.flip_triangle_edge(
                vtxs, triangles, nabrs, itri, iside
            )
        case "fortran":
            if not np.issubdtype(vtxs.dtype, np.integer):
                raise TypeError(
                    "The FORTRAN triangulation backend requires integer coordinates, "
                    + f"but got {vtxs.dtype}."
                )
            int32_info = np.iinfo(np.int32)
            if np.any(vtxs < int32_info.min) or np.any(vtxs > int32_info.max):
                raise OverflowError(
                    "The FORTRAN triangulation backend requires coordinates "
                    + "representable as int32."
                )
            if np.any(triangles >= int32_info.max):
                raise OverflowError(
                    "The FORTRAN triangulation backend requires vertex IDs "
                    + "smaller than the int32 maximum."
                )

            vtxs_f = np.asfortranarray(vtxs.T, dtype=np.int32)
            triangles_f = np.asfortranarray(triangles.T, dtype=np.int32) + 1
            nabrs_f = np.asfortranarray(nabrs.T, dtype=np.int32)
            nabrs_f[nabrs_f >= 0] += 1
            f_triangles_f, f_nabrs_f, err_code = tri_f.flip_triangle_edge(
                vtxs_f, triangles_f, nabrs_f, itri + 1, iside + 1
            )
            raise_fortran_error(
                "flip_triangle_edge", err_code, errors=_TRIANGULATION_ERRORS
            )
            f_triangles = f_triangles_f.T.astype(NpCanonIndex, order="C") - 1
            f_nabrs = f_nabrs_f.T.astype(NpCanonIndex, order="C")
            f_nabrs[f_nabrs >= 0] -= 1
        case _:
            raise ValueError(f"Unknown backend: {backend}")
    return f_triangles, f_nabrs
