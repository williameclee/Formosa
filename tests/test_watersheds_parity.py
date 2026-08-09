# Last modified
#   2026-08-03, En-Chi Lee (williameclee@gmail.com)
#     - Added test cases for function `find_acyclic_flowdirs` and
#       graph construction validity.

from tests.core import *

import pytest
import numpy as np

from formosa import D8Directions
import formosa.geomorphology.drainage.flowdir as flowdir_m
from formosa.geomorphology.drainage import watersheds as wshed_m


@pytest.fixture
def unequal_tributary_network():
    """A second-order branch joins a longer first-order branch."""
    dirs = np.zeros((4, 5), dtype=np.uint8)
    valids = np.zeros_like(dirs, dtype=bool)

    paths = {
        (0, 0): 2,  # southeast to (1, 1)
        (0, 2): 4,  # southwest to (1, 1)
        (1, 1): 3,  # south
        (2, 1): 3,  # south to the confluence
        (0, 4): 3,  # start of the longer first-order branch
        (1, 4): 4,
        (2, 3): 4,
        (3, 2): 5,
        (3, 1): 0,  # sink
    }
    for ij, direction in paths.items():
        dirs[ij] = direction
        valids[ij] = True

    expected = np.zeros_like(dirs, dtype=np.uint8)
    expected[valids] = 1
    expected[1, 1] = 2
    expected[2, 1] = 2
    expected[3, 1] = 2
    return dirs, valids, expected


@pytest.mark.parametrize("backend", BACKENDS)
def test_unequal_tributary_does_not_increase_order(unequal_tributary_network, backend):
    dirs, valids, expected = unequal_tributary_network
    dir_scheme = D8Directions(transform_codes=lambda x: x)

    orders = wshed_m.compute_flow_strahler_order(
        dirs, dir_scheme=dir_scheme, valids=valids, backend=backend
    )

    np.testing.assert_array_equal(orders, expected)


def test_strahler_backends_match_with_mask_and_supplied_indegrees(
    unequal_tributary_network,
):
    dirs, valids, _ = unequal_tributary_network
    dir_scheme = D8Directions(transform_codes=lambda x: x)
    indegs = flowdir_m.count_indegree(
        dirs, dir_scheme=dir_scheme, valids=valids, backend="python"
    )
    original_indegs = indegs.copy()

    python_orders = wshed_m.compute_flow_strahler_order(
        dirs, dir_scheme=dir_scheme, valids=valids, indegs=indegs, backend="python"
    )
    fortran_orders = wshed_m.compute_flow_strahler_order(
        dirs, dir_scheme=dir_scheme, valids=valids, indegs=indegs, backend="fortran"
    )

    np.testing.assert_array_equal(python_orders, fortran_orders)
    np.testing.assert_array_equal(indegs, original_indegs)
    assert np.all(python_orders[~valids] == 0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_masked_tributary_does_not_affect_order(backend):
    dir_scheme = D8Directions(transform_codes=lambda x: x)
    dirs = np.array(
        [
            [2, 0, 4],
            [0, 3, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )
    valids = np.array(
        [
            [T, F, F],
            [F, T, F],
            [F, T, F],
        ]
    )
    expected = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=np.uint8,
    )

    orders = wshed_m.compute_flow_strahler_order(
        dirs, dir_scheme=dir_scheme, valids=valids, backend=backend
    )

    np.testing.assert_array_equal(orders, expected)
