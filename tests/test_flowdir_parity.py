# Last modified
#   2026-08-03, En-Chi Lee (williameclee@gmail.com)
#     - Added test cases for function `find_acyclic_flowdirs`

import numpy as np
import pytest

from formosa import D8Directions
from formosa.geomorphology import flowdir


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


@pytest.mark.parametrize("backend", ["python", "fortran"])
def test_unequal_tributary_does_not_increase_order(unequal_tributary_network, backend):
    dirs, valids, expected = unequal_tributary_network
    dir_scheme = D8Directions(transform_codes=lambda x: x)

    orders = flowdir.compute_flow_strahler_order(
        dirs,
        dir_scheme=dir_scheme,
        valids=valids,
        backend=backend,
    )

    np.testing.assert_array_equal(orders, expected)


def test_strahler_backends_match_with_mask_and_supplied_indegrees(
    unequal_tributary_network,
):
    dirs, valids, _ = unequal_tributary_network
    dir_scheme = D8Directions(transform_codes=lambda x: x)
    indegs = flowdir.count_indegree(
        dirs, dir_scheme=dir_scheme, valids=valids, backend="python"
    )
    original_indegs = indegs.copy()

    python_orders = flowdir.compute_flow_strahler_order(
        dirs,
        dir_scheme=dir_scheme,
        valids=valids,
        indegs=indegs,
        backend="python",
    )
    fortran_orders = flowdir.compute_flow_strahler_order(
        dirs,
        dir_scheme=dir_scheme,
        valids=valids,
        indegs=indegs,
        backend="fortran",
    )

    np.testing.assert_array_equal(python_orders, fortran_orders)
    np.testing.assert_array_equal(indegs, original_indegs)
    assert np.all(python_orders[~valids] == 0)


@pytest.mark.parametrize("backend", ["python", "fortran"])
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
            [True, False, False],
            [False, True, False],
            [False, True, False],
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

    orders = flowdir.compute_flow_strahler_order(
        dirs,
        dir_scheme=dir_scheme,
        valids=valids,
        backend=backend,
    )

    np.testing.assert_array_equal(orders, expected)


def test_construct_flowgraph_is_backend_independent_of_masked_directions():
    dir_scheme = D8Directions(transform_codes=lambda x: x)
    dirs = np.array([[3, 3, 3], [3, 3, 3], [1, 1, 0]], dtype=np.uint8)
    valids = np.array(
        [[True, False, True], [True, True, True], [True, True, True]]
    )
    changed_dirs = dirs.copy()
    changed_dirs[~valids] = 8

    results = []
    for backend in ("python", "fortran"):
        for candidate_dirs in (dirs, changed_dirs):
            arc_orders, vertex_ijs, arc_endpts = flowdir.construct_flowgraph(
                candidate_dirs,
                dir_scheme=dir_scheme,
                valids=valids,
                min_order=1,
                backend=backend,
            )
            arcs = [
                (
                    int(order),
                    tuple(map(tuple, vertex_ijs[start : end + 1].tolist())),
                )
                for order, (start, end) in zip(arc_orders, arc_endpts)
            ]
            results.append(arcs)

    for actual in results[1:]:
        assert actual == results[0]


@pytest.mark.parametrize("backend", ["python", "fortran"])
def test_ridge_strahler_order_forwards_valid_mask(
    unequal_tributary_network, backend
):
    dirs, valids, expected = unequal_tributary_network
    dir_scheme = D8Directions(transform_codes=lambda x: x)

    orders = flowdir.compute_ridge_strahler_order(
        dirs,
        dir_scheme=dir_scheme,
        valids=valids,
        backend=backend,
        dir_is_ridge=True,
    )

    np.testing.assert_array_equal(orders, expected)


@pytest.mark.parametrize("backend", ("python", "fortran"))
def test_find_flowdir_cycles_with_feeder_and_invalid_cell(backend):
    dir_scheme = D8Directions(transform_codes=lambda x: x)
    # Cell 0 feeds the cycle between cells 1 and 2; cell 3 is invalid.
    dirs = np.array([[1, 1, 5, 0]], dtype=np.uint8)
    valids = np.array([[True, True, True, False]])

    acyclics = flowdir.find_acyclic_flowdirs(
        dirs, dir_scheme=dir_scheme, valids=valids, backend=backend
    )
    cyclics = flowdir.find_cyclic_flowdirs(
        dirs, dir_scheme=dir_scheme, valids=valids, backend=backend
    )

    np.testing.assert_array_equal(acyclics, [[True, False, False, False]])
    np.testing.assert_array_equal(cyclics, [[False, True, True, False]])


@pytest.mark.parametrize("backend", ("python", "fortran"))
def test_find_flowdir_cycles_accepts_supplied_indegrees(backend):
    dir_scheme = D8Directions(transform_codes=lambda x: x)
    dirs = np.array([[1, 1, 0]], dtype=np.uint8)
    valids = np.ones(dirs.shape, dtype=bool)
    indegs = np.array([[0, 1, 1]], dtype=np.int8)
    original_indegs = indegs.copy()

    acyclics = flowdir.find_acyclic_flowdirs(
        dirs,
        dir_scheme=dir_scheme,
        valids=valids,
        indegs=indegs,
        backend=backend,
    )

    np.testing.assert_array_equal(acyclics, valids)
    np.testing.assert_array_equal(indegs, original_indegs)


def test_find_acyclic_flowdirs_default_code_128_backend_parity():
    dirs = np.array([[0, 0], [128, 0]], dtype=np.uint8)
    valids = np.array([[False, True], [True, False]])

    python_acyclics = flowdir.find_acyclic_flowdirs(
        dirs, valids=valids, backend="python"
    )
    fortran_acyclics = flowdir.find_acyclic_flowdirs(
        dirs, valids=valids, backend="fortran"
    )

    np.testing.assert_array_equal(fortran_acyclics, python_acyclics)
    np.testing.assert_array_equal(fortran_acyclics, valids)
