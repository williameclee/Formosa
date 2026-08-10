"""
Classifies intersections between two-dimensional line segments.

Last modified: 2026-08-10, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np

from formosa.utils import Backend
from formosa.geomorphology._native import intersections as intx_f
import formosa.geomorphology.geometry._backends.intersections_py as intxs_py

import numpy.typing as npt


def _point(value: npt.ArrayLike, name: str) -> npt.NDArray[np.number]:
    """
    Returns a validated, real-valued 2D point.
    """
    point = np.asarray(value)
    if point.shape != (2,):
        raise ValueError(f"{name} must have shape (2,), " + f"but got {point.shape}.")
    if not np.issubdtype(point.dtype, np.number):
        raise TypeError(
            f"{name} must contain numeric coordinates, " + f"but got {point.dtype}."
        )
    if np.issubdtype(point.dtype, np.complexfloating):
        raise TypeError(f"{name} must contain real-valued coordinates.")
    return point


def _fortran_point(value: npt.NDArray[np.number]) -> npt.NDArray[np.float32]:
    """
    Converts a point to the representation consumed by the FORTRAN
    backend.
    """
    return np.asarray(value, dtype=np.float32, order="F")


def orient_v2(
    p1: npt.ArrayLike,
    p2: npt.ArrayLike,
    p3: npt.ArrayLike,
    backend: Backend = "fortran",
) -> int:
    """
    Computes the orientation of 3 2D points using exact comparisons.
    """
    points = (_point(p1, "p1"), _point(p2, "p2"), _point(p3, "p3"))
    match backend:
        case "python":
            return int(intxs_py.orient_v2(*points))
        case "fortran":
            return int(intx_f.orient_v2(*map(_fortran_point, points)))
        case _:
            raise ValueError(f"Unsupported backend {backend!r}.")


def on_segment(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    p: npt.ArrayLike,
    backend: Backend = "fortran",
) -> bool:
    """
    Determines whether a 2D point lies on a closed line segment.

    Parameters
    ----------
    a : NDArray[number]
        First point of the line segment.
        Must be a 2-by-0 real-valued coordinate.
    b : NDArray[number]
        Second point of the line segment.
        Must be a 2-by-0 real-valued coordinate.
    p : NDArray[number]
        Point to check.

    Returns
    -------
    flag : bool
        Whether the point lies on the line segment.
    """
    points = (_point(a, "a"), _point(b, "b"), _point(p, "p"))
    match backend:
        case "python":
            return bool(intxs_py.on_segment(*points))
        case "fortran":
            return bool(intx_f.on_segment(*map(_fortran_point, points)))
        case _:
            raise ValueError(f"Unsupported backend {backend!r}.")


def bboxes_overlap(
    l1a: npt.ArrayLike,
    l1b: npt.ArrayLike,
    l2a: npt.ArrayLike,
    l2b: npt.ArrayLike,
    backend: Backend = "fortran",
) -> bool:
    """
    Determines whether 2 closed 2D segment bounding boxes overlap.

    Parameters
    ----------
    l1a : NDArray[number]
        First point of the first line segment.
        Must be a 2-by-0 real-valued coordinate.
    l1b : NDArray[number]
        Second point of the first line segment.
        Must be a 2-by-0 real-valued coordinate.
    l2a : NDArray[number]
        First point of the second line segment.
        Must be a 2-by-0 real-valued coordinate.
    l2b : NDArray[number]
        Second point of the second line segment.
        Must be a 2-by-0 real-valued coordinate.

    Returns
    -------
    flag : bool
        Whether the bounding boxes overlap.
    """
    points = (
        _point(l1a, "l1a"),
        _point(l1b, "l1b"),
        _point(l2a, "l2a"),
        _point(l2b, "l2b"),
    )
    match backend:
        case "python":
            return bool(intxs_py.bboxes_overlap(*points))
        case "fortran":
            return bool(intx_f.bboxes_overlap(*map(_fortran_point, points)))
        case _:
            raise ValueError(f"Unsupported backend {backend!r}.")


def lines_intersect_v2(
    l1a: npt.ArrayLike,
    l1b: npt.ArrayLike,
    l2a: npt.ArrayLike,
    l2b: npt.ArrayLike,
    backend: Backend = "fortran",
) -> int:
    """
    Classifies the intersection of 2 closed 2D line segments.

    The retuned flag has the following interpretation:
      - `-1`: Disjoint segments
      - `0`: Endpoint contact
      - `1`: Interior 'X' crossing
      - `2`: Non-identical collinear overlap
      - `3`: Identical segments (orientation-agnostic)
      - `4`: Endpoint-on-interior 'T' junction
      - `5`: Degenerate segment (some line is actually a point)

    Parameters
    ----------
    l1a : NDArray[number]
        First point of the first line segment.
        Must be a 2-by-0 real-valued coordinate.
    l1b : NDArray[number]
        Second point of the first line segment.
        Must be a 2-by-0 real-valued coordinate.
    l2a : NDArray[number]
        First point of the second line segment.
        Must be a 2-by-0 real-valued coordinate.
    l2b : NDArray[number]
        Second point of the second line segment.
        Must be a 2-by-0 real-valued coordinate.

    Returns
    -------
    flag : int
        Intersection type.
    """
    points = (
        _point(l1a, "l1a"),
        _point(l1b, "l1b"),
        _point(l2a, "l2a"),
        _point(l2b, "l2b"),
    )
    match backend:
        case "python":
            return int(intxs_py.lines_intersect_v2(*points))
        case "fortran":
            return int(intx_f.lines_intersect_v2(*map(_fortran_point, points)))
        case _:
            raise ValueError(f"Unsupported backend {backend!r}.")
