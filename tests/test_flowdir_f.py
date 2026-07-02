# Last modified
#   2026-06-11, En-Chi Lee (williameclee@gmail.com)
#     - Updated function and argument names to match the standardised names
#   2026-07-01, En-Chi Lee (williameclee@gmail.com)
#     - Added test cases for `compute_downstream_indices`, `create_flowgraph`, and `compute_flow_strahler_order`
#   2026-07-02, En-Chi Lee (williameclee@gmail.com)
#     - Added test case for `compute_downstream_indices` with validity mask

import pytest
import numpy as np

from formosa import D8Directions
from formosa.geomorphology import flowdir

T = True
F = False


def test_downstreamid_3x3():
    dir_scheme = D8Directions(transform_codes=lambda x: x)

    # Config 1
    dirs = np.array(
        [
            [3, 3, 3],
            [3, 3, 3],
            [1, 1, 0],
        ]
    )

    expected_dsi = np.array(
        [
            [1, 1, 1],
            [2, 2, 2],
            [2, 2, 2],
        ]
    )
    expected_dsj = np.array(
        [
            [0, 1, 2],
            [0, 1, 2],
            [1, 2, 2],
        ]
    )
    expected_dsij = np.array(
        [
            [1, 4, 7],
            [2, 5, 8],
            [5, 8, 8],
        ]
    )
    expected_inbounds = np.array(
        [
            [T, T, T],
            [T, T, T],
            [T, T, T],
        ]
    )

    dsi, dsj, dsij, ds_inbounds = flowdir.compute_downstream_indices(
        dirs, dir_scheme=dir_scheme
    )

    np.testing.assert_array_equal(dsi, expected_dsi)
    np.testing.assert_array_equal(dsj, expected_dsj)
    np.testing.assert_array_equal(dsij, expected_dsij)
    np.testing.assert_array_equal(ds_inbounds, expected_inbounds)

    # Config 2
    dirs = np.array(
        [
            [5, 1, 1],
            [5, 1, 1],
            [5, 1, 1],
        ]
    )

    expected_dsi = np.array(
        [
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
        ]
    )
    expected_dsj = np.array(
        [
            [-1, 2, 3],
            [-1, 2, 3],
            [-1, 2, 3],
        ]
    )
    expected_dsij = np.array(
        [
            [-3, 6, 9],
            [-2, 7, 10],
            [-1, 8, 11],
        ]
    )
    expected_inbounds = np.array(
        [
            [F, T, F],
            [F, T, F],
            [F, T, F],
        ]
    )

    with pytest.warns(UserWarning):
        dsi, dsj, dsij, ds_inbounds = flowdir.compute_downstream_indices(
            dirs, dir_scheme=dir_scheme, check=False
        )

    np.testing.assert_array_equal(dsi, expected_dsi)
    np.testing.assert_array_equal(dsj, expected_dsj)
    np.testing.assert_array_equal(dsij, expected_dsij)
    np.testing.assert_array_equal(ds_inbounds, expected_inbounds)

    # Config 4 - with validity mask
    dirs = np.array(
        [
            [3, 3, 3],
            [3, 3, 3],
            [1, 1, 0],
        ]
    )
    valids = np.array([[F, T, T], [T, T, T], [T, T, T]])

    expected_dsi = np.array(
        [
            [-1, 1, 1],
            [2, 2, 2],
            [2, 2, 2],
        ]
    )
    expected_dsj = np.array(
        [
            [-1, 1, 2],
            [0, 1, 2],
            [1, 2, 2],
        ]
    )
    expected_dsij = np.array(
        [
            [-1, 4, 7],
            [2, 5, 8],
            [5, 8, 8],
        ]
    )
    expected_inbounds = np.array(
        [
            [T, T, T],
            [T, T, T],
            [T, T, T],
        ]
    )

    dsi, dsj, dsij, ds_inbounds = flowdir.compute_downstream_indices(
        dirs, dir_scheme=dir_scheme, valids=valids
    )

    np.testing.assert_array_equal(dsi, expected_dsi)
    np.testing.assert_array_equal(dsj, expected_dsj)
    np.testing.assert_array_equal(dsij, expected_dsij)
    np.testing.assert_array_equal(ds_inbounds, expected_inbounds)


def test_downstreamid_4x4():
    dir_scheme = D8Directions(transform_codes=lambda x: x)

    # Config 1
    dirs = np.array(
        [
            [1, 2, 2, 2],
            [8, 1, 1, 1],
            [8, 8, 8, 8],
            [1, 2, 1, 2],
        ]
    )
    expected_dsi = np.array(
        [
            [0, 1, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1],
            [3, 4, 3, 4],
        ]
    )
    expected_dsj = np.array(
        [
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
        ]
    )
    expected_valids = np.array(
        [
            [T, T, T, F],
            [T, T, T, F],
            [T, T, T, F],
            [T, F, T, F],
        ]
    )
    with pytest.warns(UserWarning):
        dsi, dsj, _, ds_valids = flowdir.compute_downstream_indices(
            dirs, dir_scheme=dir_scheme, check=F
        )
    np.testing.assert_array_equal(dsi, expected_dsi)
    np.testing.assert_array_equal(dsj, expected_dsj)
    np.testing.assert_array_equal(ds_valids, expected_valids)


def test_flowgraph_3x3():
    dir_scheme = D8Directions(transform_codes=lambda x: x)

    # Config 1
    dirs = np.array([[3, 3, 3], [3, 3, 3], [1, 1, 0]], dtype=np.uint8, order="F")

    expected_fg_i = np.array(
        [
            *(0, 1, np.nan, 0, 1, np.nan, 0, 1, np.nan),
            *(1, 2, np.nan, 1, 2, np.nan, 1, 2, np.nan),
            *(2, 2, np.nan, 2, 2, np.nan, 2, 2, np.nan),
        ]
    )
    expected_fg_j = np.array(
        [
            *(0, 0, np.nan, 1, 1, np.nan, 2, 2, np.nan),
            *(0, 0, np.nan, 1, 1, np.nan, 2, 2, np.nan),
            *(0, 1, np.nan, 1, 2, np.nan, 2, 2, np.nan),
        ]
    )
    fg_i, fg_j = flowdir.create_flowgraph(dirs, dir_scheme=dir_scheme)
    np.testing.assert_array_equal(fg_i, expected_fg_i)
    np.testing.assert_array_equal(fg_j, expected_fg_j)

    # Config 2
    dirs = np.array([[5, 1, 1], [5, 1, 1], [5, 1, 1]])
    expected_fg_i = np.array(
        [
            *(0, 0, np.nan),
            *(1, 1, np.nan),
            *(2, 2, np.nan),
        ]
    )
    expected_fg_j = np.array(
        [
            *(1, 2, np.nan),
            *(1, 2, np.nan),
            *(1, 2, np.nan),
        ]
    )
    with pytest.warns(UserWarning):
        fg_i, fg_j = flowdir.create_flowgraph(dirs, dir_scheme=dir_scheme)
    np.testing.assert_array_equal(fg_i, expected_fg_i)
    np.testing.assert_array_equal(fg_j, expected_fg_j)


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
        flowdir.count_indegree(dirs, dir_scheme=dir_scheme, backend="fortran"),
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

    np.testing.assert_array_equal(
        flowdir.count_indegree(dirs, dir_scheme=dir_scheme, backend="fortran"),
        expected_indegs,
    )


def test_strahler_order_3x3():
    dir_scheme = D8Directions(transform_codes=lambda x: x)

    # Config 1
    dirs = np.array(
        [
            [3, 3, 3],
            [3, 3, 3],
            [1, 1, 0],
        ]
    )

    expected_order = np.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 2, 2],
        ]
    )
    order = flowdir.compute_flow_strahler_order(dirs, dir_scheme=dir_scheme)

    np.testing.assert_array_equal(order, expected_order)

    # Config 2
    dirs = np.array(
        [
            [5, 1, 1],
            [5, 1, 1],
            [5, 1, 1],
        ]
    )

    expected_order = np.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]
    )

    order = flowdir.compute_flow_strahler_order(dirs, dir_scheme=dir_scheme)

    np.testing.assert_array_equal(order, expected_order)


def test_strahler_order_4x4():
    dir_scheme = D8Directions(transform_codes=lambda x: x)

    # Config 1
    dirs = np.array(
        [
            [1, 2, 2, 2],
            [8, 1, 1, 1],
            [8, 8, 8, 8],
            [1, 2, 1, 2],
        ]
    )
    expected_order = np.array(
        [
            [1, 2, 1, 1],
            [1, 1, 2, 2],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ]
    )
    order = flowdir.compute_flow_strahler_order(dirs, dir_scheme=dir_scheme)

    np.testing.assert_array_equal(order, expected_order)


if __name__ == "__main__":
    test_downstreamid_3x3()
    test_downstreamid_4x4()
    test_flowgraph_3x3()
    test_indegree_3x3()
    test_strahler_order_3x3()
    test_strahler_order_4x4()
