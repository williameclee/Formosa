# Last modified
#   2026-06-11, En-Chi Lee (williameclee@gmail.com)
#     - Updated function and argument names to match the standardised names
#   2026-07-01, En-Chi Lee (williameclee@gmail.com)
#     - Added test cases for `compute_downstream_indices`, `create_flowgraph`, and `compute_flow_strahler_order`
#   2026-07-02, En-Chi Lee (williameclee@gmail.com)
#     - Added test case for `compute_downstream_indices` with validity mask
#   2026-07-09, En-Chi Lee (williameclee@gmail.com)
#     - Added test case for the Fortran implementation of `construct_flowgraph`
#   2026-07-12, En-Chi Lee (williameclee@gmail.com)
#     - Added test cases for function `test_locate_invalid_graph_topogtaphy`

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


def test_network_graph_3x3():
    dir_scheme = D8Directions(transform_codes=lambda x: x)

    dirs = np.array([[3, 3, 3], [3, 3, 3], [1, 1, 0]])
    valids = np.array([[T, F, T], [T, T, T], [T, T, T]])

    exp_orders = np.array([1, 1, 1, 2])
    exp_lengths = np.array([1, 2, 3, 1])
    exp_ijs = [
        np.array([[1, 1], [2, 1]]),
        np.array([[0, 2], [1, 2], [2, 2]]),
        np.array([[0, 0], [1, 0], [2, 0], [2, 1]]),
        np.array([[2, 1], [2, 2]]),
    ]
    arc_orders, vertex_ijs, arc_endpts = flowdir.construct_flowgraph(
        dirs, dir_scheme=dir_scheme, backend="fortran", min_order=1, valids=valids
    )
    arc_lengths = arc_endpts[:, 1] - arc_endpts[:, 0]

    np.testing.assert_array_equal(arc_orders, exp_orders)
    np.testing.assert_array_equal(arc_lengths, exp_lengths)

    for i, exp_ij in enumerate(exp_ijs):
        np.testing.assert_array_equal(
            vertex_ijs[arc_endpts[i, 0] : arc_endpts[i, 1] + 1], exp_ij
        )


def test_locate_invalid_graph_topogtaphy():
    vs = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
    endpts = np.array([[0, 1], [2, 3]])
    exp_intxs = np.array([[0, 1, 0, 2, 1]], dtype=np.int32)
    intxs = flowdir.locate_invalid_graph_topology(vs, endpts, backend="fortran")
    np.testing.assert_array_equal(intxs, exp_intxs)

    vs = np.array([[0, 0], [1, 1], [2, 0], [1, 0], [0, 1]])
    endpts = np.array([[0, 2], [3, 4]])
    exp_intxs = np.array([[0, 1, 0, 3, 1]], dtype=np.int32)
    intxs = flowdir.locate_invalid_graph_topology(vs, endpts, backend="fortran")
    np.testing.assert_array_equal(intxs, exp_intxs)

    # Test self-intersection within a single arc
    vs = np.array([[0, 0], [2, 2], [2, 0], [0, 2]])
    endpts = np.array([[0, 3]])
    exp_intxs = np.array([[0, 0, 0, 2, 1]], dtype=np.int32)
    intxs = flowdir.locate_invalid_graph_topology(vs, endpts, backend="fortran")
    np.testing.assert_array_equal(intxs, exp_intxs)

    # Test no violations
    vs = np.array([[0, 0], [1, 1], [2, 2]])
    endpts = np.array([[0, 2]])
    assert flowdir.locate_invalid_graph_topology(vs, endpts, backend="fortran") is None

    # Test error handling on invalid shapes
    with pytest.raises(ValueError, match="Invalid array shapes passed"):
        flowdir.locate_invalid_graph_topology(
            np.array([1, 2, 3]), endpts, backend="fortran"
        )


def test_simplify_flowgraph():
    # 1. Single graph test
    vs = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    endpts = np.array([[0, 2]])
    simp_vs, simp_endpts, keeps = flowdir.simplify_flowgraph(
        vs, endpts, tol=1.0, check_topology=False, backend="fortran"
    )
    np.testing.assert_array_equal(keeps, [True, False, True])
    np.testing.assert_array_equal(simp_vs, [[0.0, 0.0], [2.0, 2.0]])
    np.testing.assert_array_equal(simp_endpts, [[0, 1]])

    # 2. Multiple graph test (list of standard arrays)
    vs0 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    endpts0 = np.array([[0, 2]])
    vs1 = np.array([[3.0, 3.0], [4.0, 4.0], [5.0, 5.0]])
    endpts1 = np.array([[0, 2]])

    simp_vs_list, simp_endpts_list, keeps_list = flowdir.simplify_flowgraph(
        [vs0, vs1], [endpts0, endpts1], tol=1.0, check_topology=False, backend="fortran"
    )
    assert isinstance(simp_vs_list, list)
    np.testing.assert_array_equal(keeps_list[0], [True, False, True])
    np.testing.assert_array_equal(keeps_list[1], [True, False, True])
    np.testing.assert_array_equal(simp_vs_list[0], [[0.0, 0.0], [2.0, 2.0]])
    np.testing.assert_array_equal(simp_vs_list[1], [[3.0, 3.0], [5.0, 5.0]])
    np.testing.assert_array_equal(simp_endpts_list[0], [[0, 1]])
    np.testing.assert_array_equal(simp_endpts_list[1], [[0, 1]])

    # 3. Multiple graph test (tuple of transposed/differing shapes)
    vs0_t = vs0.T  # shape (2, 3)
    endpts0_t = endpts0.T  # shape (2, 1)

    simp_vs_tuple, simp_endpts_tuple, keeps_tuple = flowdir.simplify_flowgraph(
        (vs0_t, vs1), (endpts0_t, endpts1), tol=1.0, check_topology=False, backend="fortran"
    )
    assert isinstance(simp_vs_tuple, tuple)
    np.testing.assert_array_equal(keeps_tuple[0], [True, False, True])
    np.testing.assert_array_equal(keeps_tuple[1], [True, False, True])
    # Verify orientation restoration
    assert simp_vs_tuple[0].shape == (2, 2)
    assert simp_endpts_tuple[0].shape == (2, 1)
    assert simp_vs_tuple[1].shape == (2, 2)
    assert simp_endpts_tuple[1].shape == (1, 2)

    # 4. Test topology correction (single graph)
    vs_topo = np.array([[0.0, 0.8], [1.0, 2.0], [2.0, 0.2], [0.5, 0.5], [1.5, 0.5]])
    endpts_topo = np.array([[0, 2], [3, 4]])

    # Under tol = 1.5 and check_topology = False, simplification occurs and causes intersection
    _, _, keeps_no_check = flowdir.simplify_flowgraph(
        vs_topo, endpts_topo, tol=1.5, check_topology=False, backend="fortran"
    )
    # Vertex 1 should be removed
    np.testing.assert_array_equal(keeps_no_check, [True, False, True, True, True])

    # Under tol = 1.5 and check_topology = True, it detects intersection, reduces tolerance,
    # and keeps Vertex 1 to avoid intersection
    _, _, keeps_with_check = flowdir.simplify_flowgraph(
        vs_topo, endpts_topo, tol=1.5, check_topology=True, backend="fortran"
    )
    # Vertex 1 should be kept
    np.testing.assert_array_equal(keeps_with_check, [True, True, True, True, True])


if __name__ == "__main__":
    test_downstreamid_3x3()
    test_downstreamid_4x4()
    test_flowgraph_3x3()
    test_indegree_3x3()
    test_strahler_order_3x3()
    test_strahler_order_4x4()
    test_network_graph_3x3()
    test_locate_invalid_graph_topogtaphy()
    test_simplify_flowgraph()
