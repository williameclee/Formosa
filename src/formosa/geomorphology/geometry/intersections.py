"""
Evaluates two-dimensional geometric predicates.

This module exposes the public orientation, in-circle, and
line-segment intersection API. It dispatches calculations to the
internal Python or FORTRAN backend and selects C-interoperable
integer overloads where possible.

Last modified: 2026-08-12, En-Chi Lee (williameclee@gmail.com)
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


def _dint_dtype(points: tuple[NDArray[NpReal], ...]) -> np.dtype | None:
    """
    Selects a lossless C-interoperable integer input kind, if
    possible.
    """
    if not all(np.issubdtype(point.dtype, np.integer) for point in points):
        return None

    minimum = min(int(np.min(point)) for point in points)
    maximum = max(int(np.max(point)) for point in points)
    if np.iinfo(np.int32).min <= minimum and maximum <= np.iinfo(np.int32).max:
        return np.dtype(np.int32)
    if np.iinfo(np.int64).min <= minimum and maximum <= np.iinfo(np.int64).max:
        return np.dtype(np.int64)
    return None


def _fint_points(
    points: tuple[NDArray[NpReal], ...], dtype: np.dtype
) -> tuple[NDArray[np.integer], ...]:
    """
    Converts integer points to a C-interoperable FORTRAN
    representation.
    """
    return tuple(np.asarray(point, dtype=dtype, order="F") for point in points)


def orient_v2(
    p1: ArrayLike, p2: ArrayLike, p3: ArrayLike, backend: Backend = "fortran"
) -> Real:
    """
    Calculates the signed orientation determinant of 3 2D points.

    A positive result indicates a counterclockwise turn, a negative
    result indicates a clockwise turn, and zero indicates
    collinearity.

    Parameters
    ----------
    p1 : ArrayLike
        First real-valued coordinate with shape `(2,)`.
    p2 : ArrayLike
        Second real-valued coordinate with shape `(2,)`.
    p3 : ArrayLike
        Third real-valued coordinate with shape `(2,)`.
    backend : {"fortran", "python"}, optional
        Computational backend.
        Default backend is `"fortran"`.

    Returns
    -------
    det : int | float
        Signed orientation determinant.

        All-integer inputs return an integer when a C-interoperable
        integer kind can represent the coordinates. Other inputs
        return a float.

    Raises
    ------
    ValueError
        If a point does not have shape `(2,)` or the backend is
        unsupported.
    TypeError
        If a point contains non-numeric or complex-valued
        coordinates.

    Notes
    -----
    The Python backend uses exact integer arithmetic. The FORTRAN
    backend uses double-precision intermediates for integer inputs
    and same-kind saturating results. If a 32-bit result saturates,
    this wrapper retries the calculation using the 64-bit overload.
    The anticipated coordinate magnitudes of up to approximately
    100000 are represented exactly in double precision; integer
    coordinates above `2**53` may not be.
    """
    points = (_point(p1, "p1"), _point(p2, "p2"), _point(p3, "p3"))
    match backend:
        case "python":
            return intxs_py.orient_v2(*points)
        case "fortran":
            match _dint_dtype(points):
                case dtype if dtype == np.dtype(np.int32):
                    det = intx_f.orient_v2_int32(
                        *_fint_points(points, dtype)  # type: ignore
                    )
                    if det in (np.iinfo(np.int32).min, np.iinfo(np.int32).max):
                        det = intx_f.orient_v2_int64(
                            *_fint_points(points, np.dtype(np.int64))
                        )
                    return det
                case dtype if dtype == np.dtype(np.int64):
                    return intx_f.orient_v2_int64(*_fint_points(points, dtype))  # type: ignore
                case _:
                    return float(intx_f.orient_v2_real(*map(_fortran_point, points)))
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

    For counterclockwise triangle vertices, a positive result places
    `p` inside their circumcircle, zero places it on the circle, and
    a negative result places it outside. Reversing the triangle
    orientation reverses the determinant sign.

    When `oriented=True`, the result is normalised so that the
    classification is independent of the order of the triangle
    vertices.

    Parameters
    ----------
    a : ArrayLike
        First real-valued triangle coordinate with shape `(2,)`.
    b : ArrayLike
        Second real-valued triangle coordinate with shape `(2,)`.
    c : ArrayLike
        Third real-valued triangle coordinate with shape `(2,)`.
    p : ArrayLike
        Real-valued test coordinate with shape `(2,)`.
    oriented : bool, optional
        Whether to normalise the determinant sign for the
        orientation of `a`, `b`, and `c`.
        Default option is `False`.
    backend : {"fortran", "python"}, optional
        Computational backend.
        Default backend is `"fortran"`.

    Returns
    -------
    det : int | float
        Signed in-circle determinant.

        All-integer inputs return an integer when a C-interoperable
        integer kind can represent the coordinates. Other inputs
        return a float.

    Raises
    ------
    ValueError
        If a point does not have shape `(2,)`, the backend is
        unsupported, or sign normalisation is requested for a
        collinear triangle.
    TypeError
        If a point contains non-numeric or complex-valued
        coordinates.

    Notes
    -----
    The Python backend uses exact integer arithmetic. The FORTRAN
    backend uses double-precision intermediates for integer inputs
    and same-kind saturating results. If a 32-bit result saturates,
    this wrapper retries the calculation using the 64-bit overload.
    The anticipated coordinate magnitudes of up to approximately
    100000 are represented exactly in double precision; integer
    coordinates above `2**53` may not be.
    """
    points = (
        _point(a, "a"),
        _point(b, "b"),
        _point(c, "c"),
        _point(p, "p"),
    )
    match backend:
        case "python":
            det = intxs_py.incircle(*points)
        case "fortran":
            match _dint_dtype(points):
                case dtype if dtype == np.dtype(np.int32):
                    det = intx_f.incircle_int32(*_fint_points(points, dtype))  # type: ignore
                    if det in (np.iinfo(np.int32).min, np.iinfo(np.int32).max):
                        det = intx_f.incircle_int64(
                            *_fint_points(points, np.dtype(np.int64))
                        )
                    det = det
                case dtype if dtype == np.dtype(np.int64):
                    det = intx_f.incircle_int64(*_fint_points(points, dtype))  # type: ignore
                case _:
                    det = float(intx_f.incircle_real(*map(_fortran_point, points)))
        case _:
            raise ValueError(f"Unsupported backend {backend!r}.")

    if not oriented:
        return det

    orient = orient_v2(a, b, c, backend=backend)
    if orient == 0:
        raise ValueError(
            "Cannot normalise an in-circle determinant for a collinear triangle."
        )
    return det if orient > 0 else -det


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
    backend : {"fortran", "python"}, optional
            Computational backend.
            Default backend is `"fortran"`.

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
