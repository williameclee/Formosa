"""
Parity tests of the classification of line segment intersections.

Last modified: 2026-08-09, En-Chi Lee (williameclee@gmail.com)
"""

import pytest
import numpy as np

import formosa.geomorphology.geometry.intersections as intx_m

LINE_INTERSECTION_CASES = [
    # Parallel lines
    ((0, 0), (0, 1), (1, 0), (1, 1), -1),
    ((0, 0), (3, 3), (1, 2), (0, 3), -1),
    ((0, 0), (1, 0), (2, 0), (3, 0), -1),
    # Sharing endpoints
    ((0, 0), (1, 0), (1, 0), (1, 1), 0),
    ((0, 0), (1, 0), (1, 0), (2, 0), 0),
    ((0, 0), (-2, -2), (3, 1), (-2, -2), 0),
    # Crossing
    ((0, 0), (1, 1), (1, 0), (0, 1), 1),
    ((0, 0), (3, 3), (1, 2), (3, 0), 1),
    # Collinear overlapping lines
    ((0, 0), (0, 2), (0, 1), (0, 3), 2),
    ((0, 0), (4, 0), (1, 0), (3, 0), 2),
    ((0, 0), (2, 2), (1, 1), (3, 3), 2),
    # Collinear overlapping lines, sharing endpoints
    ((0, 0), (0, 2), (0, 1), (0, 2), 2),
    ((0, 0), (3, 3), (2, 2), (3, 3), 2),
    # Identical lines
    ((0, 0), (0, 1), (0, 0), (0, 1), 3),
    ((2, 5), (4, 3), (4, 3), (2, 5), 3),
    # T junction
    ((0, 0), (2, 0), (1, 1), (1, 0), 4),
    ((-1, -1), (3, 1), (1, 0), (5, 7), 4),
    # degenerate segment (some line is actually a point)
    ((0, 0), (0, 0), (0, 0), (1, 1), 5),
    ((0, 0), (0, 0), (1, 1), (1, 1), 5),
]


@pytest.mark.parametrize(
    ("l1a", "l1b", "l2a", "l2b", "expected"), LINE_INTERSECTION_CASES
)
def test_intersection_parity(l1a, l1b, l2a, l2b, expected):
    python_flag: int = intx_m.lines_intersect_v2(l1a, l1b, l2a, l2b, backend="python")
    fortran_flag = intx_m.lines_intersect_v2(l1a, l1b, l2a, l2b, backend="fortran")

    assert python_flag == expected
    assert fortran_flag == python_flag


@pytest.mark.parametrize(
    ("l1a", "l1b", "l2a", "l2b", "expected"), LINE_INTERSECTION_CASES
)
def test_intersection_parity_symmetry(l1a, l1b, l2a, l2b, expected):
    python_flag: int = intx_m.lines_intersect_v2(l2a, l2b, l1a, l1b, backend="python")
    fortran_flag = intx_m.lines_intersect_v2(l2a, l2b, l1a, l1b, backend="fortran")

    assert python_flag == expected
    assert fortran_flag == python_flag


@pytest.mark.parametrize("backend", ["python", "fortran"])
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
