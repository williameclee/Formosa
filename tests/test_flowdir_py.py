"""
Tests flow-direction derivation using the Python backend.

Last modified: 2026-08-10, En-Chi Lee (williameclee@gmail.com)
"""

import pytest
import numpy as np

from formosa import D8Directions
import formosa.geomorphology.drainage.flowdir as flowdir_m


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
        flowdir_m.count_indegree(dirs, dir_scheme=dir_scheme, backend="python"),
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
        indegs = flowdir_m.count_indegree(dirs, dir_scheme=dir_scheme, backend="python")

    np.testing.assert_array_equal(indegs, expected_indegs)


@pytest.mark.parametrize("name", ("valids", "indegs"))
def test_find_acyclic_flowdirs_rejects_shape_mismatch(name):
    kwargs = {name: np.ones((2, 1), dtype=bool)}
    with pytest.raises(ValueError, match="Shapes"):
        flowdir_m.find_acyclic_flowdirs(
            np.zeros((1, 1), dtype=np.uint8), backend="python", **kwargs  # type: ignore
        )
