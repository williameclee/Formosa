# Last modified
#   2026-07-02, En-Chi Lee (williameclee@gmail.com)
#     - Added test cases for the Python implementation of `count_indegree`

import pytest
import numpy as np

from formosa import D8Directions
from formosa.geomorphology import flowdir

T = True
F = False


def test_indegree_3x3():
    dir_scheme = D8Directions(transform_codes=lambda x: x)

    # Config 1
    dirs = np.array([[3, 3, 3], [3, 3, 3], [1, 1, 0]])

    expected_indegs = np.array(
        [
            [0, 0, 0],
            [1, 1, 1],
            [1, 2, 2],
        ]
    )

    np.testing.assert_array_equal(
        flowdir.count_indegree(dirs, dir_scheme=dir_scheme, backend="python"),
        expected_indegs,
    )

    # Config 2
    dirs = np.array([[5, 1, 1], [5, 1, 1], [5, 1, 1]])

    expected_indegs = np.array(
        [
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
        ]
    )
    with pytest.warns(UserWarning):
        indegs = flowdir.count_indegree(dirs, dir_scheme=dir_scheme, backend="python")

    np.testing.assert_array_equal(indegs, expected_indegs)


if __name__ == "__main__":
    test_indegree_3x3()
