"""
Verifies line-segment intersection parity across configured
backends.

Last modified: 2026-08-11, En-Chi Lee (williameclee@gmail.com)
"""

import pytest
import numpy as np

from formosa.utils import BACKENDS
import formosa.geomorphology.geometry.intersections as intx_m
from formosa.geomorphology.geometry.intersections import IntersectionKind


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    ("l1a", "l1b", "l2a", "l2b", "exp_flag"),
    [
        # Parallel lines
        ((0, 0), (0, 1), (1, 0), (1, 1), IntersectionKind.DISJOINT_SEGMENTS),
        ((0, 0), (3, 3), (1, 2), (0, 3), IntersectionKind.DISJOINT_SEGMENTS),
        ((0, 0), (1, 0), (2, 0), (3, 0), IntersectionKind.DISJOINT_SEGMENTS),
        # Sharing endpoints
        ((0, 0), (1, 0), (1, 0), (1, 1), IntersectionKind.ENDPOINT_CONTACT),
        ((0, 0), (1, 0), (1, 0), (2, 0), IntersectionKind.ENDPOINT_CONTACT),
        ((0, 0), (-2, -2), (3, 1), (-2, -2), IntersectionKind.ENDPOINT_CONTACT),
        # Crossing
        ((0, 0), (1, 1), (1, 0), (0, 1), IntersectionKind.INTERIOR_CROSSING),
        ((0, 0), (3, 3), (1, 2), (3, 0), IntersectionKind.INTERIOR_CROSSING),
        # Collinear overlapping lines
        ((0, 0), (0, 2), (0, 1), (0, 3), IntersectionKind.COLLINEAR_OVERLAP),
        ((0, 0), (4, 0), (1, 0), (3, 0), IntersectionKind.COLLINEAR_OVERLAP),
        ((0, 0), (2, 2), (1, 1), (3, 3), IntersectionKind.COLLINEAR_OVERLAP),
        # Collinear overlapping lines, sharing endpoints
        ((0, 0), (0, 2), (0, 1), (0, 2), IntersectionKind.COLLINEAR_OVERLAP),
        ((0, 0), (3, 3), (2, 2), (3, 3), IntersectionKind.COLLINEAR_OVERLAP),
        # Identical lines
        ((0, 0), (0, 1), (0, 0), (0, 1), IntersectionKind.IDENTICAL_SEGMENTS),
        ((2, 5), (4, 3), (4, 3), (2, 5), IntersectionKind.IDENTICAL_SEGMENTS),
        # T junction
        ((0, 0), (2, 0), (1, 1), (1, 0), IntersectionKind.T_JUNCTION),
        ((-1, -1), (3, 1), (1, 0), (5, 7), IntersectionKind.T_JUNCTION),
        # degenerate segment (some line is actually a point)
        ((0, 0), (0, 0), (0, 0), (1, 1), IntersectionKind.DEGENERATE_SEGMENT),
        ((0, 0), (0, 0), (1, 1), (1, 1), IntersectionKind.DEGENERATE_SEGMENT),
    ],
)
def test_intersection_parity(l1a, l1b, l2a, l2b, exp_flag, backend):
    flag: int = intx_m.lines_intersect_v2(l1a, l1b, l2a, l2b, backend=backend)
    assert flag == exp_flag
    # Flip the segments; result should be the same
    flag: int = intx_m.lines_intersect_v2(l2a, l2b, l1a, l1b, backend=backend)
    assert flag == exp_flag


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    ("function", "args", "expected"),
    [
        (intx_m.orient_v2, ((0, 0), (1, 0), (1, 1)), 1),
        (intx_m.on_segment, ((0, 0), (2, 0), (1, 0)), True),
        (
            intx_m.bboxes_overlap,
            ((0, 0), (2, 2), (1, 1), (3, 3)),
            True,
        ),
        (
            intx_m.lines_intersect_v2,
            ((0, 0), (1, 1), (1, 0), (0, 1)),
            1,
        ),
    ],
)
def test_public_wrappers_select_backend(backend, function, args, expected):
    result = function(*args, backend=backend)

    assert result == expected
    assert type(result) is type(expected)


def test_public_wrapper_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unsupported backend"):
        intx_m.orient_v2((0, 0), (1, 0), (1, 1), backend="unknown")  # type: ignore


@pytest.mark.parametrize(
    ("point", "error"),
    [
        ((0, 1, 2), ValueError),
        (("x", "y"), TypeError),
        (np.array([1 + 2j, 3 + 4j]), TypeError),
    ],
)
def test_public_wrapper_validates_points(point, error):
    with pytest.raises(error):
        intx_m.orient_v2(point, (1, 0), (1, 1), backend="python")
