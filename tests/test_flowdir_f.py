import pytest
import numpy as np

from formosa import D8Directions


def test_indegree_3x3():
    from formosa.geomorphology import flowdir

    dir_scheme = D8Directions(transform_codes=lambda x: x)

    # Config 1
    dirs = np.array([[3, 3, 3], [3, 3, 3], [1, 1, 0]], dtype=np.uint8, order="F")

    expected_indegree = np.array(
        [
            [0, 0, 0],
            [1, 1, 1],
            [1, 2, 2],
        ],
        dtype=np.int8,
        order="F",
    )

    np.testing.assert_array_equal(
        flowdir.compute_indegree(dirs, dir_scheme=dir_scheme, backend="fortran"),
        expected_indegree,
    )

    # Config 2
    dirs = np.array([[5, 1, 1], [5, 1, 1], [5, 1, 1]], dtype=np.uint8, order="F")

    expected_indegree = np.array(
        [
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
        ],
        dtype=np.int8,
        order="F",
    )

    np.testing.assert_array_equal(
        flowdir.compute_indegree(dirs, dir_scheme=dir_scheme, backend="fortran"),
        expected_indegree,
    )


if __name__ == "__main__":
    test_indegree_3x3()
