# Last modified
#   2026-07-02, En-Chi Lee (williameclee@gmail.com)
#     - Added test cases for the Python implementation of `count_indegree`
#   2026-07-09, En-Chi Lee (williameclee@gmail.com)
#     - Added test case for the Python implementation of `construct_flowgraph` and function `concat_flowgraph`
#   2026-07-12, En-Chi Lee (williameclee@gmail.com)
#     - Added test cases for function `locate_invalid_graph_topogtaphy`
#   2026-07-14, En-Chi Lee (williameclee@gmail.com)
#     - Updated `geomorphology.flowdir` to the new submodule name
#   2026-07-27, En-Chi Lee (williameclee@gmail.com)
#     - Added test cases for function `insert_endpt`


import pytest
import numpy as np

from formosa import D8Directions
import formosa.geomorphology.flowdir as flowdir

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
        dirs, dir_scheme=dir_scheme, backend="python", min_order=1, valids=valids
    )
    arc_lengths = arc_endpts[:, 1] - arc_endpts[:, 0]

    np.testing.assert_array_equal(arc_orders, exp_orders)
    np.testing.assert_array_equal(arc_lengths, exp_lengths)

    for i, exp_ij in enumerate(exp_ijs):
        np.testing.assert_array_equal(
            vertex_ijs[arc_endpts[i, 0] : arc_endpts[i, 1] + 1], exp_ij
        )


def test_network_graph_concat_3x3():
    dir_scheme = D8Directions(transform_codes=lambda x: x)

    dirs = np.array([[3, 3, 3], [3, 3, 3], [1, 1, 0]])
    valids = np.array([[T, F, T], [T, T, T], [T, T, T]])
    arc_orders, vertex_ijs, arc_endpts = flowdir.construct_flowgraph(
        dirs, dir_scheme=dir_scheme, backend="python", min_order=1, valids=valids
    )

    exp_s_orders = np.array([1, 2])
    exp_s_endpts = np.array([[0, 10], [12, 13]])

    s_arc_orders, s_vertex_ijs, s_arc_endpts = flowdir.concat_flowgraph(
        arc_orders, vertex_ijs, arc_endpts
    )
    assert s_vertex_ijs.shape[0] == vertex_ijs.shape[0] + arc_orders.shape[0] - 1
    assert arc_endpts[-1, 1] == vertex_ijs.shape[0] - 1
    np.testing.assert_array_equal(s_arc_orders, exp_s_orders)
    np.testing.assert_array_equal(s_arc_endpts, exp_s_endpts)


def test_locate_invalid_graph_topogtaphy():
    vs = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
    endpts = np.array([[0, 1], [2, 3]])
    exp_intxs = np.array([[0, 1, 0, 2, 1]], dtype=np.int32)
    intxs = flowdir.locate_invalid_graph_topology(vs, endpts, backend="python")
    np.testing.assert_array_equal(intxs, exp_intxs)

    vs = np.array([[0, 0], [1, 1], [2, 0], [1, 0], [0, 1]])
    endpts = np.array([[0, 2], [3, 4]])
    exp_intxs = np.array([[0, 1, 0, 3, 1]], dtype=np.int32)
    intxs = flowdir.locate_invalid_graph_topology(vs, endpts, backend="python")
    np.testing.assert_array_equal(intxs, exp_intxs)

    # Test self-intersection within a single arc
    vs = np.array([[0, 0], [2, 2], [2, 0], [0, 2]])
    endpts = np.array([[0, 3]])
    exp_intxs = np.array([[0, 0, 0, 2, 1]], dtype=np.int32)
    intxs = flowdir.locate_invalid_graph_topology(vs, endpts, backend="python")
    np.testing.assert_array_equal(intxs, exp_intxs)

    # Test no violations
    vs = np.array([[0, 0], [1, 1], [2, 2]])
    endpts = np.array([[0, 2]])
    assert flowdir.locate_invalid_graph_topology(vs, endpts, backend="python") is None

    # Test error handling on invalid shapes
    with pytest.raises(ValueError, match="Invalid array shapes passed"):
        flowdir.locate_invalid_graph_topology(
            np.array([1, 2, 3]), endpts, backend="python"
        )


def test_graph_insert_endpt():
    from formosa.geomorphology.flowdir.graphs import insert_endpt

    ijs = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
    endpts = np.array([[0, 2], [3, 4]])
    orders = np.array([1, 2])

    # Non-existent additional endpoint
    with pytest.warns():
        o_orders, o_ijs, o_endpts = insert_endpt(orders, ijs, endpts, np.array([1, 0]))
    np.testing.assert_array_equal(ijs, o_ijs)
    np.testing.assert_array_equal(endpts, o_endpts)
    np.testing.assert_array_equal(orders, o_orders)

    # Additional endpoint is already an endpoint
    o_orders, o_ijs, o_endpts = insert_endpt(orders, ijs, endpts, np.array([0, 1]))
    np.testing.assert_array_equal(ijs, o_ijs)
    np.testing.assert_array_equal(endpts, o_endpts)
    np.testing.assert_array_equal(orders, o_orders)
    o_orders, o_ijs, o_endpts = insert_endpt(orders, ijs, endpts, np.array([4, 5]))
    np.testing.assert_array_equal(ijs, o_ijs)
    np.testing.assert_array_equal(endpts, o_endpts)
    np.testing.assert_array_equal(orders, o_orders)
    o_orders, o_ijs, o_endpts = insert_endpt(orders, ijs, endpts, np.array([6, 7]))
    np.testing.assert_array_equal(ijs, o_ijs)
    np.testing.assert_array_equal(endpts, o_endpts)
    np.testing.assert_array_equal(orders, o_orders)
    o_orders, o_ijs, o_endpts = insert_endpt(orders, ijs, endpts, np.array([8, 9]))
    np.testing.assert_array_equal(ijs, o_ijs)
    np.testing.assert_array_equal(endpts, o_endpts)
    np.testing.assert_array_equal(orders, o_orders)

    # Additional endpoint is already an endpoint, specified as index
    o_orders, o_ijs, o_endpts = insert_endpt(orders, ijs, endpts, 0)
    np.testing.assert_array_equal(ijs, o_ijs)
    np.testing.assert_array_equal(endpts, o_endpts)
    np.testing.assert_array_equal(orders, o_orders)
    o_orders, o_ijs, o_endpts = insert_endpt(orders, ijs, endpts, 2)
    np.testing.assert_array_equal(ijs, o_ijs)
    np.testing.assert_array_equal(endpts, o_endpts)
    np.testing.assert_array_equal(orders, o_orders)

    # Insert endpoint
    o_orders, o_ijs, o_endpts = insert_endpt(orders, ijs, endpts, np.array([2, 3]))
    exp_ijs = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [2, 3], [4, 5]])
    exp_endpts = np.array([[0, 1], [3, 4], [5, 6]])
    exp_orders = np.array([1, 2, 1])
    np.testing.assert_array_equal(o_ijs, exp_ijs)
    np.testing.assert_array_equal(o_endpts, exp_endpts)
    np.testing.assert_array_equal(o_orders, exp_orders)

    # Insert endpoint, specified as index
    o_orders, o_ijs, o_endpts = insert_endpt(orders, ijs, endpts, 1)
    exp_ijs = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [2, 3], [4, 5]])
    exp_endpts = np.array([[0, 1], [3, 4], [5, 6]])
    exp_orders = np.array([1, 2, 1])
    np.testing.assert_array_equal(o_ijs, exp_ijs)
    np.testing.assert_array_equal(o_endpts, exp_endpts)
    np.testing.assert_array_equal(o_orders, exp_orders)

    # Invalid input shapes
    with pytest.raises(AssertionError):
        o_orders, o_ijs, o_endpts = insert_endpt(
            orders, ijs, endpts[:-1, :], np.array([2, 3])
        )


if __name__ == "__main__":
    test_indegree_3x3()
    test_network_graph_3x3()
    test_network_graph_concat_3x3()
    test_locate_invalid_graph_topogtaphy()
    test_graph_insert_endpt()
