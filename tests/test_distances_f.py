# Last modified
#   2026-07-10, En-Chi Lee (williameclee@gmail.com)
#     - Added test cases for the FORTRAN function of `lines_intersect_v2`

import pytest

from formosa.geomorphology.flowdir_f import distances as dist_f


def test_intersection_detection():
    # Parallel lines
    assert (dist_f.lines_intersect_v2([0, 0], [0, 1], [1, 0], [1, 1])) == -1
    assert (dist_f.lines_intersect_v2([0, 0], [3, 3], [1, 2], [0, 3])) == -1
    # Sharing endpoints
    assert dist_f.lines_intersect_v2([0, 0], [1, 0], [1, 0], [1, 1]) == 0
    assert dist_f.lines_intersect_v2([0, 0], [-2, -2], [3, 1], [-2, -2]) == 0
    # Crossing
    assert dist_f.lines_intersect_v2([0, 0], [1, 1], [1, 0], [0, 1]) == 1
    assert dist_f.lines_intersect_v2([0, 0], [3, 3], [1, 2], [3, 0]) == 1
    # Collinear overlapping lines
    assert dist_f.lines_intersect_v2([0, 0], [0, 2], [0, 1], [0, 3]) == 2
    assert dist_f.lines_intersect_v2([0, 0], [2, 2], [1, 1], [3, 3]) == 2
    # Collinear overlapping lines, sharing endpoints
    assert dist_f.lines_intersect_v2([0, 0], [0, 2], [0, 1], [0, 2]) == 2
    assert dist_f.lines_intersect_v2([0, 0], [3, 3], [2, 2], [3, 3]) == 2
    # Identical lines
    assert (dist_f.lines_intersect_v2([0, 0], [0, 1], [0, 0], [0, 1])) == 3
    assert (dist_f.lines_intersect_v2([2, 5], [4, 3], [4, 3], [2, 5])) == 3
    # T junction
    assert dist_f.lines_intersect_v2([0, 0], [2, 0], [1, 1], [1, 0]) == 4
    assert dist_f.lines_intersect_v2([-1, -1], [3, 1], [1, 0], [5, 7]) == 4


if __name__ == "__main__":
    test_intersection_detection()
