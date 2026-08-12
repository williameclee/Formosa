"""
Provides the public API for unconstrained Delaunay triangulation.

This module dispatches to the Python or FORTRAN backend and
normalises native inputs, outputs, and errors.

Created: 2026-08-12, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np

from formosa.geomorphology._native import meshing_triangulation as tri_f
import formosa.geomorphology.meshing._backends.triangulation_py as tri_py
from formosa.geomorphology.drainage.network import GraphTopologyError

from numpy.typing import NDArray
from formosa.utils import Backend, raise_fortran_error
from formosa.utils.typing import NpCoords, NpCanonIndex

_TRIANGULATION_ERRORS = {
    1: (ValueError, "invalid triangulation input"),
    2: (MemoryError, "unable to allocate triangulation workspace"),
    3: (RuntimeError, "triangulation capacity exceeded"),
    4: (GraphTopologyError, "point set did not produce a valid triangulation"),
}


def triangulate_points(
    vtxs: NDArray[NpCoords], backend: Backend = "fortran"
) -> NDArray[NpCanonIndex]:
    """
    Computes an unconstrained Delaunay triangulation of 2D points.

    The returned triangle vertex IDs are zero-based for both
    backends. The native backend currently accepts only coordinates
    representable as signed 32-bit integers.

    Parameters
    ----------
    vtxs : NDArray[int]
        Unique point coordinates.
        At least 3 points are required.
    backend : {"python", "fortran"}, optional
        Triangulation implementation to use.

    Returns
    -------
    NDArray[int32], shape (F, 3)
        Counterclockwise triangle vertex IDs.
    """
    vtxs = np.asarray(vtxs)
    tri_py._validate_triangulate_points(vtxs)

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
    return triangles
