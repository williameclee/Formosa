"""
Verifies flow-metric parity between the Python and Fortran backends.

Last modified: 2026-08-24, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np
import pytest
from numpy.typing import NDArray

import formosa.geomorphology.drainage.flowdir as flowdir_m
import formosa.geomorphology.drainage.metrics as metrics_m
from formosa import D8Directions
from formosa.utils import BACKENDS, Backend, NpFlowDir
from tests.core import *


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    ("dirs", "expected_orders", "should_warn"),
    [
        (
            [[3, 3, 3], [3, 3, 3], [1, 1, 0]],
            [[1, 1, 1], [1, 1, 1], [1, 2, 2]],
            False,
        ),
        (
            [[5, 1, 1], [5, 1, 1], [5, 1, 1]],
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            True,
        ),
        (
            [[1, 2, 2, 2], [8, 1, 1, 1], [8, 8, 8, 8], [1, 2, 1, 2]],
            [[1, 2, 1, 1], [1, 1, 2, 2], [1, 1, 1, 1], [1, 1, 1, 1]],
            True,
        ),
    ],
)
def test_strahler_order_reference_cases(
    backend: Backend,
    dirs: NDArray[NpFlowDir],
    expected_orders: NDArray[np.integer],
    should_warn: bool,
):
    dir_scheme = D8Directions(transform_codes=lambda x: x)

    if should_warn and backend == "python":
        with pytest.warns(UserWarning):
            orders = metrics_m.compute_flow_strahler_order(
                np.array(dirs), dir_scheme=dir_scheme, backend=backend
            )
    else:
        orders = metrics_m.compute_flow_strahler_order(
            np.array(dirs), dir_scheme=dir_scheme, backend=backend
        )

    np.testing.assert_array_equal(orders, np.array(expected_orders))


@pytest.fixture
def unequal_tributary_network() -> tuple[
    NDArray[NpFlowDir], NDArray[np.bool_], NDArray[np.uint8]
]:
    """A second-order branch joins a longer first-order branch."""
    dirs = np.zeros((4, 5), dtype=NpFlowDir)
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
def test_unequal_tributary_does_not_increase_order(
    unequal_tributary_network, backend: Backend
):
    dirs, valids, expected = unequal_tributary_network
    dir_scheme = D8Directions(transform_codes=lambda x: x)

    orders = metrics_m.compute_flow_strahler_order(
        dirs, dir_scheme=dir_scheme, valids=valids, backend=backend
    )

    np.testing.assert_array_equal(orders, expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_strahler_with_mask_and_supplied_indegrees(
    unequal_tributary_network, backend: Backend
):
    dirs, valids, expected = unequal_tributary_network
    dir_scheme = D8Directions(transform_codes=lambda x: x)
    indegs = flowdir_m.count_indegree(dirs, dir_scheme, valids=valids, backend="python")
    original_indegs = indegs.copy()

    orders = metrics_m.compute_flow_strahler_order(
        dirs, dir_scheme, valids=valids, indegs=indegs, backend=backend
    )

    np.testing.assert_array_equal(orders, expected)
    np.testing.assert_array_equal(indegs, original_indegs)
    assert np.all(orders[~valids] == 0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_masked_tributary_does_not_affect_order(backend: Backend):
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

    orders = metrics_m.compute_flow_strahler_order(
        dirs, dir_scheme=dir_scheme, valids=valids, backend=backend
    )

    np.testing.assert_array_equal(orders, expected)
