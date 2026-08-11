"""
Classifies intersections between two-dimensional line segments.

Last modified: 2026-08-11, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np

from formosa.utils import Backend
from formosa.geomorphology._native import intersections as intx_f
from formosa.geomorphology.geometry._backends.intersections_py import IntersectionKind
import formosa.geomorphology.geometry._backends.intersections_py as intxs_py

from numpy.typing import NDArray, ArrayLike
from formosa.utils.typing import Real, NpReal


def _point(value: ArrayLike, name: str) -> NDArray[NpReal]:
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


def _fortran_point(value: NDArray[NpReal]) -> NDArray[np.float32]:
    """
    Converts a point to the representation consumed by the FORTRAN
    backend.
    """
    return np.asarray(value, dtype=np.float32, order="F")


def orient_v2(
    p1: ArrayLike, p2: ArrayLike, p3: ArrayLike, backend: Backend = "fortran"
) -> Real:
    """
    Returns the signed determinant of three two-dimensional points.

    A positive result indicates a counterclockwise turn, a negative result a
    clockwise turn, and zero collinearity. Integer inputs use exact arithmetic
    with the Python backend.
    """
    points = (_point(p1, "p1"), _point(p2, "p2"), _point(p3, "p3"))
    match backend:
        case "python":
            return intxs_py.orient_v2(*points)
        case "fortran":
            return float(intx_f.orient_v2(*map(_fortran_point, points)))
        case _:
            raise ValueError(f"Unsupported backend {backend!r}.")


def incircle(
    a: ArrayLike,
    b: ArrayLike,
    c: ArrayLike,
    p: ArrayLike,
    oriented: bool = False,
    backend: Backend = "fortran",
) -> Real:
    """
    Calculates the signed in-circle determinant for 4 2D points.

    - Positive determinant means `p` lies inside their circumcircle;
    - 0 means it is cocircular;
    - Negative determinant means it lies outside.

    When `oriented=Dalse`, the sign of the determinant depends on 
    the orientation of the triangle. For counterclockwise triangle
    vertices `a`, `b`, and `c`, the sign is the same as described
    above. Reversing the triangle orientation reverses the sign.

    Parameters
    ----------
    a : NDArray[number]
        First point of the triangle.
        Must be a 2-by-0 real-valued coordinate.
    b : NDArray[number]
        Second point of the triangle.
        Must be a 2-by-0 real-valued coordinate.
    c : NDArray[number]
        Third point of the triangle.
        Must be a 2-by-0 real-valued coordinate.
    p : NDArray[number]
        Point to test against the triangle's circumcircle.
        Must be a 2-by-0 real-valued coordinate.
    oriented : bool, optional
        Whether to normalise the determinant sign for the
        orientation of `a`, `b`, and `c`.
        A collinear triangle raises `ValueError` in this mode.
        Default option is `False`.

    Returns
    -------
    det : int | float
        Signed in-circle determinant.

    Notes
    -----
    Integer inputs use exact arithmetic with the Python backend.
    """
    points = (
        _point(a, "a"),
        _point(b, "b"),
        _point(c, "c"),
        _point(p, "p"),
    )
    match backend:
        case "python":
            determinant = intxs_py.incircle(*points)
        case "fortran":
            determinant = float(intx_f.incircle(*map(_fortran_point, points)))
        case _:
            raise ValueError(f"Unsupported backend {backend!r}.")

    if not oriented:
        return determinant

    orientation = orient_v2(a, b, c, backend=backend)
    if orientation == 0:
        raise ValueError(
            "Cannot normalise an in-circle determinant for a collinear triangle."
        )
    return determinant if orientation > 0 else -determinant


def on_segment(
    a: ArrayLike, b: ArrayLike, p: ArrayLike, backend: Backend = "fortran"
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
    l1a: ArrayLike,
    l1b: ArrayLike,
    l2a: ArrayLike,
    l2b: ArrayLike,
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
    l1a: ArrayLike,
    l1b: ArrayLike,
    l2a: ArrayLike,
    l2b: ArrayLike,
    backend: Backend = "fortran",
) -> int:
    """
    Classifies the intersection of 2 closed 2D line segments.

    The return flag has the following interpretation (as in
    :class:`IntersectionKind`):
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
