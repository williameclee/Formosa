import pytest

from formosa.geomorphology.flowdir_f import distances as dist_f
from formosa.geomorphology.distances_py import _lines_intersect_v2

LINE_INTERSECTION_CASES = [
    ((0, 0), (0, 1), (1, 0), (1, 1), -1),
    ((0, 0), (3, 3), (1, 2), (0, 3), -1),
    ((0, 0), (1, 0), (2, 0), (3, 0), -1),
    ((0, 0), (1, 0), (1, 0), (1, 1), 0),
    ((0, 0), (1, 0), (1, 0), (2, 0), 0),
    ((0, 0), (1, 1), (1, 0), (0, 1), 1),
    ((0, 0), (3, 3), (1, 2), (3, 0), 1),
    ((0, 0), (0, 2), (0, 1), (0, 3), 2),
    ((0, 0), (4, 0), (1, 0), (3, 0), 2),
    ((0, 0), (0, 1), (0, 0), (0, 1), 3),
    ((2, 5), (4, 3), (4, 3), (2, 5), 3),
    ((0, 0), (2, 0), (1, 1), (1, 0), 4),
    ((-1, -1), (3, 1), (1, 0), (5, 7), 4),
    ((0, 0), (0, 0), (0, 0), (1, 1), 5),
    ((0, 0), (0, 0), (1, 1), (1, 1), 5),
]


@pytest.mark.parametrize(
    ("l1a", "l1b", "l2a", "l2b", "expected"), LINE_INTERSECTION_CASES
)
def test_python_fortran_intersection_parity(l1a, l1b, l2a, l2b, expected):
    python_flag = _lines_intersect_v2(l1a, l1b, l2a, l2b)
    fortran_flag = dist_f.lines_intersect_v2(l1a, l1b, l2a, l2b)

    assert python_flag == expected
    assert fortran_flag == python_flag
