"""
Classifies line-segment intersections using the Python backend.

This module implements internal routines called by the public-facing
geometry API and is not intended to be used directly.

Last modified: 2026-08-17, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np
from enum import IntFlag

from typing import overload
from numpy.typing import NDArray
from formosa.utils.typing import Real, NpReal


@overload
def orient(
    p1: NDArray[np.integer], p2: NDArray[np.integer], p3: NDArray[np.integer]
) -> int: ...


@overload
def orient(
    p1: NDArray[np.floating], p2: NDArray[np.floating], p3: NDArray[np.floating]
) -> float: ...


@overload
def orient(p1: NDArray[NpReal], p2: NDArray[NpReal], p3: NDArray[NpReal]) -> Real: ...


def orient(p1: NDArray[NpReal], p2: NDArray[NpReal], p3: NDArray[NpReal]) -> Real:
    """
    Computes the signed determinant of three two-dimensional points.
    """
    p1x, p1y = p1.tolist()
    p2x, p2y = p2.tolist()
    p3x, p3y = p3.tolist()
    return (p2x - p1x) * (p3y - p1y) - (p2y - p1y) * (p3x - p1x)


@overload
def incircle(
    a: NDArray[np.integer],
    b: NDArray[np.integer],
    c: NDArray[np.integer],
    p: NDArray[np.integer],
) -> int: ...


@overload
def incircle(
    a: NDArray[np.floating],
    b: NDArray[np.floating],
    c: NDArray[np.floating],
    p: NDArray[np.floating],
) -> float: ...


@overload
def incircle(
    a: NDArray[NpReal], b: NDArray[NpReal], c: NDArray[NpReal], p: NDArray[NpReal]
) -> Real: ...


def incircle(
    a: NDArray[NpReal], b: NDArray[NpReal], c: NDArray[NpReal], p: NDArray[NpReal]
) -> Real:
    """
    Calculates the signed in-circle determinant for 4 2D points.
    """
    ax, ay = a.tolist()
    bx, by = b.tolist()
    cx, cy = c.tolist()
    px, py = p.tolist()

    adx, ady = ax - px, ay - py
    bdx, bdy = bx - px, by - py
    cdx, cdy = cx - px, cy - py
    abdet = adx * bdy - bdx * ady
    bcdet = bdx * cdy - cdx * bdy
    cadet = cdx * ady - adx * cdy
    alift = adx * adx + ady * ady
    blift = bdx * bdx + bdy * bdy
    clift = cdx * cdx + cdy * cdy
    return alift * bcdet + blift * cadet + clift * abdet


def on_segment(a: NDArray[NpReal], b: NDArray[NpReal], p: NDArray[NpReal]) -> bool:
    """
    Determines whether a 2D point lies on a closed line segment.
    """
    if orient(a, b, p) != 0:
        return False
    return bool(
        p[0] >= min(a[0], b[0])
        and p[0] <= max(a[0], b[0])
        and p[1] >= min(a[1], b[1])
        and p[1] <= max(a[1], b[1])
    )


def bboxes_overlap(
    p1: NDArray[NpReal], p2: NDArray[NpReal], p3: NDArray[NpReal], p4: NDArray[NpReal]
) -> bool:
    """
    Determines whether 2 closed 2D segment bounding boxes overlap.
    """
    return bool(
        max(min(p1[0], p2[0]), min(p3[0], p4[0]))
        <= min(max(p1[0], p2[0]), max(p3[0], p4[0]))
        and max(min(p1[1], p2[1]), min(p3[1], p4[1]))
        <= min(max(p1[1], p2[1]), max(p3[1], p4[1]))
    )


class IntersectionKind(IntFlag):
    DISJOINT_SEGMENTS = -1
    ENDPOINT_CONTACT = 0
    INTERIOR_CROSSING = 1
    COLLINEAR_OVERLAP = 2
    IDENTICAL_SEGMENTS = 3
    T_JUNCTION = 4
    DEGENERATE_SEGMENT = 5


def lines_intersect(
    l1a: NDArray[NpReal],
    l1b: NDArray[NpReal],
    l2a: NDArray[NpReal],
    l2b: NDArray[NpReal],
) -> IntersectionKind:
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
    """
    l1a = np.asarray(l1a)
    l1b = np.asarray(l1b)
    l2a = np.asarray(l2a)
    l2b = np.asarray(l2b)

    if np.array_equal(l1a, l1b) or np.array_equal(l2a, l2b):
        return IntersectionKind.DEGENERATE_SEGMENT
    if not bboxes_overlap(l1a, l1b, l2a, l2b):
        return IntersectionKind.DISJOINT_SEGMENTS

    eq_l1al2a = np.array_equal(l1a, l2a)
    eq_l1al2b = np.array_equal(l1a, l2b)
    eq_l1bl2a = np.array_equal(l1b, l2a)
    eq_l1bl2b = np.array_equal(l1b, l2b)
    if (eq_l1al2a and eq_l1bl2b) or (eq_l1al2b and eq_l1bl2a):
        return IntersectionKind.IDENTICAL_SEGMENTS

    o1 = orient(l1a, l1b, l2a)
    o2 = orient(l1a, l1b, l2b)
    o3 = orient(l2a, l2b, l1a)
    o4 = orient(l2a, l2b, l1b)
    opposite_12 = (o1 < 0 and o2 > 0) or (o1 > 0 and o2 < 0)
    opposite_34 = (o3 < 0 and o4 > 0) or (o3 > 0 and o4 < 0)
    if opposite_12 and opposite_34:
        return IntersectionKind.INTERIOR_CROSSING

    if o1 == 0 and o2 == 0 and o3 == 0 and o4 == 0:
        if abs(l1b[0] - l1a[0]) >= abs(l1b[1] - l1a[1]):
            a0, a1 = sorted((l1a[0], l1b[0]))
            c0, c1 = sorted((l2a[0], l2b[0]))
        else:
            a0, a1 = sorted((l1a[1], l1b[1]))
            c0, c1 = sorted((l2a[1], l2b[1]))

        overlap0 = max(a0, c0)
        overlap1 = min(a1, c1)
        if overlap1 < overlap0:
            return IntersectionKind.DISJOINT_SEGMENTS
        if overlap1 <= overlap0:
            return IntersectionKind.ENDPOINT_CONTACT
        return IntersectionKind.COLLINEAR_OVERLAP

    if eq_l1al2a or eq_l1al2b or eq_l1bl2a or eq_l1bl2b:
        return IntersectionKind.ENDPOINT_CONTACT

    if (
        on_segment(l1a, l1b, l2a)
        or on_segment(l1a, l1b, l2b)
        or on_segment(l2a, l2b, l1a)
        or on_segment(l2a, l2b, l1b)
    ):
        return IntersectionKind.T_JUNCTION
    return IntersectionKind.DISJOINT_SEGMENTS
