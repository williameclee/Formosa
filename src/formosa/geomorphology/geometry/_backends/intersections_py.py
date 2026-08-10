"""
Classifies line-segment intersections using the Python backend.

This module implements internal routines called by the public-facing
geometry API and is not intended to be used directly.

Last modified: 2026-08-10, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np

import numpy.typing as npt


def orient_v2(
    p1: npt.NDArray[np.number],
    p2: npt.NDArray[np.number],
    p3: npt.NDArray[np.number],
) -> int:
    """
    Computes the orientation of 3 2D points using exact comparisons.
    """
    xprod = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    if xprod == 0:
        return 0
    if xprod < 0:
        return -1
    return 1


def on_segment(
    a: npt.NDArray[np.number], b: npt.NDArray[np.number], p: npt.NDArray[np.number]
) -> bool:
    """
    Determines whether a 2D point lies on a closed line segment.
    """
    if orient_v2(a, b, p) != 0:
        return False
    return bool(
        p[0] >= min(a[0], b[0])
        and p[0] <= max(a[0], b[0])
        and p[1] >= min(a[1], b[1])
        and p[1] <= max(a[1], b[1])
    )


def bboxes_overlap(
    p1: npt.NDArray[np.number],
    p2: npt.NDArray[np.number],
    p3: npt.NDArray[np.number],
    p4: npt.NDArray[np.number],
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


def lines_intersect_v2(
    l1a: npt.NDArray[np.number],
    l1b: npt.NDArray[np.number],
    l2a: npt.NDArray[np.number],
    l2b: npt.NDArray[np.number],
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
    """
    l1a = np.asarray(l1a)
    l1b = np.asarray(l1b)
    l2a = np.asarray(l2a)
    l2b = np.asarray(l2b)

    if np.array_equal(l1a, l1b) or np.array_equal(l2a, l2b):
        return 5
    if not bboxes_overlap(l1a, l1b, l2a, l2b):
        return -1

    eq_l1al2a = np.array_equal(l1a, l2a)
    eq_l1al2b = np.array_equal(l1a, l2b)
    eq_l1bl2a = np.array_equal(l1b, l2a)
    eq_l1bl2b = np.array_equal(l1b, l2b)
    if (eq_l1al2a and eq_l1bl2b) or (eq_l1al2b and eq_l1bl2a):
        return 3

    o1 = orient_v2(l1a, l1b, l2a)
    o2 = orient_v2(l1a, l1b, l2b)
    o3 = orient_v2(l2a, l2b, l1a)
    o4 = orient_v2(l2a, l2b, l1b)
    if o1 * o2 < 0 and o3 * o4 < 0:
        return 1

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
            return -1
        if overlap1 <= overlap0:
            return 0
        return 2

    if eq_l1al2a or eq_l1al2b or eq_l1bl2a or eq_l1bl2b:
        return 0

    if (
        on_segment(l1a, l1b, l2a)
        or on_segment(l1a, l1b, l2b)
        or on_segment(l2a, l2b, l1a)
        or on_segment(l2a, l2b, l1b)
    ):
        return 4
    return -1
