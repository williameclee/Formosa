from tests.core import *

import pytest
import numpy as np

from formosa import D8Directions
import formosa.geomorphology.drainage.network.construction as constr_m
import formosa.geomorphology.drainage.network.editing as editing_m


def test_remove_unused_vertices_compacts_arc_ranges():
    vertices = np.array(
        [
            [99, 99],
            *([0, 0], [1, 0]),
            [98, 98],
            [97, 97],
            *([2, 0], [3, 0], [4, 0]),
            [96, 96],
        ]
    )
    endpts = np.array([[1, 2], [5, 7]], dtype=np.int32)

    compact_vertices, compact_endpts = editing_m.remove_unused_vertices(
        vertices, endpts
    )

    np.testing.assert_array_equal(
        compact_vertices, [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]]
    )
    np.testing.assert_array_equal(compact_endpts, [[0, 1], [2, 4]])
    assert compact_endpts[0, 1] + 1 == compact_endpts[1, 0]


def test_remove_unused_vertices_handles_empty_graph():
    vertices = np.array([[99, 99]], dtype=np.int32)
    endpts = np.empty((0, 2), dtype=np.int32)

    compact_vertices, compact_endpts = editing_m.remove_unused_vertices(
        vertices, endpts
    )

    assert compact_vertices.shape == (0, 2)
    assert compact_vertices.dtype == vertices.dtype
    assert compact_endpts.shape == (0, 2)
    assert compact_endpts.dtype == endpts.dtype


def test_insert_endpt_can_remove_unused_vertices():
    orders = np.array([1], dtype=np.int8)
    vertices = np.array([[99, 99], [0, 0], [1, 0], [2, 0], [98, 98]])
    endpts = np.array([[1, 3]], dtype=np.int32)

    out_orders, out_vertices, out_endpts = editing_m.insert_endpt(
        orders, vertices, endpts, 2, remove_unused=True
    )

    np.testing.assert_array_equal(out_orders, [1, 1])
    np.testing.assert_array_equal(out_vertices, [[0, 0], [1, 0], [1, 0], [2, 0]])
    np.testing.assert_array_equal(out_endpts, [[0, 1], [2, 3]])


def test_graph_insert_endpt():
    ijs = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
    endpts = np.array([[0, 2], [3, 4]])
    orders = np.array([1, 2])

    # Non-existent additional endpoint
    with pytest.warns():
        o_orders, o_ijs, o_endpts = editing_m.insert_endpt(
            orders, ijs, endpts, np.array([1, 0])
        )
    np.testing.assert_array_equal(ijs, o_ijs)
    np.testing.assert_array_equal(endpts, o_endpts)
    np.testing.assert_array_equal(orders, o_orders)

    # Additional endpoint is already an endpoint
    o_orders, o_ijs, o_endpts = editing_m.insert_endpt(
        orders, ijs, endpts, np.array([0, 1])
    )
    np.testing.assert_array_equal(ijs, o_ijs)
    np.testing.assert_array_equal(endpts, o_endpts)
    np.testing.assert_array_equal(orders, o_orders)
    o_orders, o_ijs, o_endpts = editing_m.insert_endpt(
        orders, ijs, endpts, np.array([4, 5])
    )
    np.testing.assert_array_equal(ijs, o_ijs)
    np.testing.assert_array_equal(endpts, o_endpts)
    np.testing.assert_array_equal(orders, o_orders)
    o_orders, o_ijs, o_endpts = editing_m.insert_endpt(
        orders, ijs, endpts, np.array([6, 7])
    )
    np.testing.assert_array_equal(ijs, o_ijs)
    np.testing.assert_array_equal(endpts, o_endpts)
    np.testing.assert_array_equal(orders, o_orders)
    o_orders, o_ijs, o_endpts = editing_m.insert_endpt(
        orders, ijs, endpts, np.array([8, 9])
    )
    np.testing.assert_array_equal(ijs, o_ijs)
    np.testing.assert_array_equal(endpts, o_endpts)
    np.testing.assert_array_equal(orders, o_orders)

    # Additional endpoint is already an endpoint, specified as index
    o_orders, o_ijs, o_endpts = editing_m.insert_endpt(orders, ijs, endpts, 0)
    np.testing.assert_array_equal(ijs, o_ijs)
    np.testing.assert_array_equal(endpts, o_endpts)
    np.testing.assert_array_equal(orders, o_orders)
    o_orders, o_ijs, o_endpts = editing_m.insert_endpt(orders, ijs, endpts, 2)
    np.testing.assert_array_equal(ijs, o_ijs)
    np.testing.assert_array_equal(endpts, o_endpts)
    np.testing.assert_array_equal(orders, o_orders)

    # Insert endpoint
    o_orders, o_ijs, o_endpts = editing_m.insert_endpt(
        orders, ijs, endpts, np.array([2, 3])
    )
    exp_ijs = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [2, 3], [4, 5]])
    exp_endpts = np.array([[0, 1], [3, 4], [5, 6]])
    exp_orders = np.array([1, 2, 1])
    np.testing.assert_array_equal(o_ijs, exp_ijs)
    np.testing.assert_array_equal(o_endpts, exp_endpts)
    np.testing.assert_array_equal(o_orders, exp_orders)

    # Insert endpoint, specified as index
    o_orders, o_ijs, o_endpts = editing_m.insert_endpt(orders, ijs, endpts, 1)
    exp_ijs = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [2, 3], [4, 5]])
    exp_endpts = np.array([[0, 1], [3, 4], [5, 6]])
    exp_orders = np.array([1, 2, 1])
    np.testing.assert_array_equal(o_ijs, exp_ijs)
    np.testing.assert_array_equal(o_endpts, exp_endpts)
    np.testing.assert_array_equal(o_orders, exp_orders)

    # Invalid input shapes
    with pytest.raises(AssertionError):
        o_orders, o_ijs, o_endpts = editing_m.insert_endpt(
            orders, ijs, endpts[:-1, :], np.array([2, 3])
        )

    # Multiple occurrences of a same vertex - is an endpoint
    ijs = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [4, 5], [10, 11]])
    endpts = np.array([[0, 2], [3, 4], [5, 6]])
    orders = np.array([1, 2, 3])
    o_orders, o_ijs, o_endpts = editing_m.insert_endpt(
        orders, ijs, endpts, np.array([4, 5])
    )
    np.testing.assert_array_equal(ijs, o_ijs)
    np.testing.assert_array_equal(endpts, o_endpts)
    np.testing.assert_array_equal(orders, o_orders)

    # Multiple occurrences of a same vertex - is both an endpoint and an interior vertex (should not actually happen normally)
    ijs = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [2, 3], [10, 11]])
    endpts = np.array([[0, 2], [3, 4], [5, 6]])
    orders = np.array([1, 2, 3])

    o_orders, o_ijs, o_endpts = editing_m.insert_endpt(
        orders, ijs, endpts, np.array([2, 3])
    )
    exp_ijs = np.array(
        [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [2, 3], [10, 11], [2, 3], [4, 5]]
    )
    exp_endpts = np.array([[0, 1], [3, 4], [5, 6], [7, 8]])
    exp_orders = np.array([1, 2, 3, 1])

    np.testing.assert_array_equal(o_ijs, exp_ijs)
    np.testing.assert_array_equal(o_endpts, exp_endpts)
    np.testing.assert_array_equal(o_orders, exp_orders)

    # Multiple occurrences of a same vertex - is an interior vertex (should not actually happen normally)
    ijs = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [2, 3], [12, 13]])
    endpts = np.array([[0, 2], [3, 4], [5, 7]])
    orders = np.array([1, 2, 3])

    o_orders, o_ijs, o_endpts = editing_m.insert_endpt(
        orders, ijs, endpts, np.array([2, 3])
    )
    exp_ijs = np.array(
        [
            *([0, 1], [2, 3], [4, 5]),
            *([6, 7], [8, 9]),
            *([10, 11], [2, 3], [12, 13]),
            *([2, 3], [4, 5]),
            *([2, 3], [12, 13]),
        ]
    )
    exp_endpts = np.array([[0, 1], [3, 4], [5, 6], [8, 9], [10, 11]])
    exp_orders = np.array([1, 2, 3, 1, 3])

    np.testing.assert_array_equal(o_ijs, exp_ijs)
    np.testing.assert_array_equal(o_endpts, exp_endpts)
    np.testing.assert_array_equal(o_orders, exp_orders)


def test_network_graph_concat_3x3():
    dir_scheme = D8Directions(transform_codes=lambda x: x)

    dirs = np.array([[3, 3, 3], [3, 3, 3], [1, 1, 0]])
    valids = np.array([[T, F, T], [T, T, T], [T, T, T]])
    arc_orders, vertex_ijs, arc_endpts = constr_m.construct_flowgraph(
        dirs, dir_scheme=dir_scheme, backend="python", min_order=1, valids=valids
    )

    exp_s_orders = np.array([1, 2])
    exp_s_endpts = np.array([[0, 10], [12, 13]])

    s_arc_orders, s_vertex_ijs, s_arc_endpts = editing_m.concat_flowgraph(
        arc_orders, vertex_ijs, arc_endpts
    )
    assert s_vertex_ijs.shape[0] == vertex_ijs.shape[0] + arc_orders.shape[0] - 1
    assert arc_endpts[-1, 1] == vertex_ijs.shape[0] - 1
    np.testing.assert_array_equal(s_arc_orders, exp_s_orders)
    np.testing.assert_array_equal(s_arc_endpts, exp_s_endpts)
