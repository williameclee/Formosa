# Last modified
#   2026-08-08, En-Chi Lee (williameclee@gmail.com)
#     - Aggregated all intersection-related tests and made both
#       backends tested at the same time

import pytest

from formosa.geomorphology.flowdir_f import intersections as intx_f
import formosa.geomorphology.geometry.intersections_py as intx_py

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
    python_flag: int = intx_py.lines_intersect_v2(l1a, l1b, l2a, l2b)
    fortran_flag = intx_f.lines_intersect_v2(l1a, l1b, l2a, l2b)

    assert python_flag == expected
    assert fortran_flag == python_flag


@pytest.mark.parametrize(
    ("l1a", "l1b", "l2a", "l2b", "expected"), LINE_INTERSECTION_CASES
)
def test_intersection_parity_symmetry(l1a, l1b, l2a, l2b, expected):
    python_flag: int = intx_py.lines_intersect_v2(l2a, l2b, l1a, l1b)
    fortran_flag = intx_f.lines_intersect_v2(l2a, l2b, l1a, l1b)

    assert python_flag == expected
    assert fortran_flag == python_flag
