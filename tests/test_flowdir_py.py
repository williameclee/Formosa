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
#   2026-07-28, En-Chi Lee (williameclee@gmail.com)
#     - Added test cases for functions `find_graph_overlaps` and `solve_graph_overlaps`
#   2026-08-03, En-Chi Lee (williameclee@gmail.com)
#     - Added test cases for function `find_acyclic_flowdirs` and graph construction validity


import pytest
import numpy as np

from formosa import D8Directions
import formosa.geomorphology.flowdir as flowdir
from formosa.geomorphology.flowdir.graphs import graphs as graphs_module
from formosa.geomorphology.flowdir.graphs import graphs_py as graphs_py_module

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


def test_construct_flowgraph_rejects_incomplete_backend_output(monkeypatch):
    dir_scheme = D8Directions(transform_codes=lambda x: x)
    dirs = np.array([[1, 0]], dtype=np.uint8)
    orders = np.ones(dirs.shape, dtype=np.uint8)

    def omit_selected_edge(*args, **kwargs):
        return (
            0,
            0,
            np.empty((0,), dtype=np.int8),
            np.empty((2, 0), dtype=np.int32),
            np.empty((2, 0), dtype=np.int32),
        )

    monkeypatch.setattr(
        graphs_py_module,
        "_construct_flowgraph_py",
        omit_selected_edge,
    )

    with pytest.raises(flowdir.IncompleteFlowGraphError) as exc_info:
        graphs_module.construct_flowgraph(
            dirs,
            dir_scheme=dir_scheme,
            orders=orders,
            min_order=1,
            backend="python",
        )

    np.testing.assert_array_equal(exc_info.value.missing_ijs, [[0, 0], [0, 1]])


def test_construct_flowgraph_rejects_missing_directed_edge(monkeypatch):
    dir_scheme = D8Directions(transform_codes=lambda x: x)
    dirs = np.array([[1, 1, 0]], dtype=np.uint8)
    orders = np.ones(dirs.shape, dtype=np.uint8)

    def omit_second_edge(*args, **kwargs):
        # All expected cells are present in the raw vertex buffer, but the only
        # returned arc stops at (0, 1); the edge (0, 1) -> (0, 2) is omitted.
        return (
            1,
            3,
            np.array([1], dtype=np.int8),
            np.array([[0, 0, 0], [0, 1, 2]], dtype=np.int32),
            np.array([[0], [1]], dtype=np.int32),
        )

    monkeypatch.setattr(
        graphs_py_module,
        "_construct_flowgraph_py",
        omit_second_edge,
    )

    with pytest.raises(flowdir.IncompleteFlowGraphError) as exc_info:
        graphs_module.construct_flowgraph(
            dirs,
            dir_scheme=dir_scheme,
            orders=orders,
            min_order=1,
            backend="python",
        )

    np.testing.assert_array_equal(
        exc_info.value.missing_edges,
        [[0, 1, 0, 2]],
    )
    np.testing.assert_array_equal(
        exc_info.value.missing_ijs,
        [[0, 1], [0, 2]],
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

    # Multiple occurrences of a same vertex - is an endpoint
    ijs = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [4, 5], [10, 11]])
    endpts = np.array([[0, 2], [3, 4], [5, 6]])
    orders = np.array([1, 2, 3])
    o_orders, o_ijs, o_endpts = insert_endpt(orders, ijs, endpts, np.array([4, 5]))
    np.testing.assert_array_equal(ijs, o_ijs)
    np.testing.assert_array_equal(endpts, o_endpts)
    np.testing.assert_array_equal(orders, o_orders)

    # Multiple occurrences of a same vertex - is both an endpoint and an interior vertex (should not actually happen normally)
    ijs = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [2, 3], [10, 11]])
    endpts = np.array([[0, 2], [3, 4], [5, 6]])
    orders = np.array([1, 2, 3])

    o_orders, o_ijs, o_endpts = insert_endpt(orders, ijs, endpts, np.array([2, 3]))
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

    o_orders, o_ijs, o_endpts = insert_endpt(orders, ijs, endpts, np.array([2, 3]))
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


def test_find_graph_overlaps():
    from formosa.geomorphology.flowdir.graphs.graphs import find_graph_overlaps

    # Include matching stored vertices outside the ranges referenced by the arcs
    g1_ijs = np.array(
        [
            [99, 99],
            *([0, 0], [1, 0], [2, 0]),
            *([3, 0], [4, 0], [5, 0]),
            [98, 98],
        ]
    )
    g1_endpts = np.array([[1, 3], [4, 6]])
    g2_ijs = np.array(
        [
            *([0, 0], [2, 0], [8, 8]),
            *([1, 0], [4, 0], [9, 9]),
            [99, 99],
        ]
    )
    g2_endpts = np.array([[0, 2], [3, 5]])

    vert_vert, intr_intr, g1_intr_g2_vert, g1_vert_g2_intr = find_graph_overlaps(
        g1_ijs, g1_endpts, g2_ijs, g2_endpts
    )

    np.testing.assert_array_equal(vert_vert, np.array([[0, 0]]))
    np.testing.assert_array_equal(intr_intr, np.array([[4, 0]]))
    np.testing.assert_array_equal(g1_intr_g2_vert, np.array([[1, 0]]))
    np.testing.assert_array_equal(g1_vert_g2_intr, np.array([[2, 0]]))

    # The matching unused coordinate must not be reported as an overlap
    for overlap_type in (
        vert_vert,
        intr_intr,
        g1_intr_g2_vert,
        g1_vert_g2_intr,
    ):
        assert not np.any(np.all(overlap_type == [99, 99], axis=1))


def test_find_graph_overlaps_is_symmetric_with_repeated_coordinates():
    from formosa.geomorphology.flowdir.graphs.graphs import find_graph_overlaps

    g1_ijs = np.array([[0, 0], [1, 0], [2, 0], [1, 0], [3, 0]])
    g1_endpts = np.array([[0, 2], [3, 4]])
    g2_ijs = np.array([[0, 0], [2, 0], [4, 0], [1, 0], [5, 0]])
    g2_endpts = np.array([[0, 2], [3, 4]])

    forward = find_graph_overlaps(g1_ijs, g1_endpts, g2_ijs, g2_endpts)
    forward = find_graph_overlaps(g1_ijs, g1_endpts, g2_ijs, g2_endpts)
    reverse = find_graph_overlaps(g2_ijs, g2_endpts, g1_ijs, g1_endpts)

    np.testing.assert_array_equal(forward[0], reverse[0])
    np.testing.assert_array_equal(forward[1], reverse[1])
    np.testing.assert_array_equal(forward[2], reverse[3])
    np.testing.assert_array_equal(forward[3], reverse[2])
    # A coordinate is classified as an endpoint when any used occurrence is
    # an endpoint, even if another occurrence is interior.
    assert np.any(np.all(forward[0] == [1, 0], axis=1))


@pytest.mark.parametrize("empty_graph", (1, 2))
def test_find_graph_overlaps_accepts_an_empty_graph(empty_graph):
    from formosa.geomorphology.flowdir.graphs.graphs import find_graph_overlaps

    empty_ijs = np.empty((0, 2), dtype=np.int32)
    empty_endpts = np.empty((0, 2), dtype=np.int32)
    ijs = np.array([[0, 0], [1, 0]], dtype=np.int32)
    endpts = np.array([[0, 1]], dtype=np.int32)
    graphs = ((empty_ijs, empty_endpts), (ijs, endpts))
    if empty_graph == 2:
        graphs = graphs[::-1]

    overlaps = find_graph_overlaps(*graphs[0], *graphs[1])

    for overlap_type in overlaps:
        assert overlap_type.shape == (0, 2)
        assert overlap_type.dtype == np.int32


@pytest.mark.parametrize(
    (
        "allows_arcs_overlap",
        "expected_narcs",
        "expected_vert_vert",
        "expected_intr_intr",
    ),
    [
        (True, 4, np.array([[1, 0], [3, 0]]), np.array([[2, 0]])),
        (
            False,
            5,
            np.array([[1, 0], [2, 0], [3, 0]]),
            np.empty((0, 2), dtype=int),
        ),
    ],
)
def test_solve_graph_overlaps(
    allows_arcs_overlap,
    expected_narcs,
    expected_vert_vert,
    expected_intr_intr,
):
    from formosa.geomorphology.flowdir.graphs.graphs import (
        find_graph_overlaps,
        solve_graph_overlaps,
    )

    # The graphs share three consecutive interior vertices in opposite directions
    g1_orders = np.array([1, 9])
    g1_ijs = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [10, 0], [11, 0]])
    g1_endpts = np.array([[0, 4], [5, 6]])
    g2_orders = np.array([2, 9])
    g2_ijs = np.array([[5, 0], [3, 0], [2, 0], [1, 0], [6, 0], [10, 1], [11, 1]])
    g2_endpts = np.array([[0, 4], [5, 6]])

    (
        solved_g1_orders,
        solved_g1_ijs,
        solved_g1_endpts,
        solved_g2_orders,
        solved_g2_ijs,
        solved_g2_endpts,
    ) = solve_graph_overlaps(
        g1_orders,
        g1_ijs,
        g1_endpts,
        g2_orders,
        g2_ijs,
        g2_endpts,
        allows_arcs_overlap=allows_arcs_overlap,
    )

    assert solved_g1_orders.size == expected_narcs
    assert solved_g2_orders.size == expected_narcs
    assert solved_g1_endpts.shape == (expected_narcs, 2)
    assert solved_g2_endpts.shape == (expected_narcs, 2)

    vert_vert, intr_intr, g1_intr_g2_vert, g1_vert_g2_intr = find_graph_overlaps(
        solved_g1_ijs,
        solved_g1_endpts,
        solved_g2_ijs,
        solved_g2_endpts,
    )
    np.testing.assert_array_equal(vert_vert, expected_vert_vert)
    np.testing.assert_array_equal(intr_intr, expected_intr_intr)
    assert g1_intr_g2_vert.size == 0
    assert g1_vert_g2_intr.size == 0


def test_solve_graph_overlaps_with_repeated_coordinates():
    from formosa.geomorphology.flowdir.graphs.graphs import (
        find_graph_overlaps,
        solve_graph_overlaps,
    )

    # Coordinate (2, 0) occurs in two arcs of the first graph
    g1_orders = np.array([1, 2])
    g1_ijs = np.array(
        [
            [0, 0],
            [1, 0],
            [2, 0],
            [3, 0],
            [4, 0],
            [8, 0],
            [2, 0],
            [9, 0],
        ]
    )
    g1_endpts = np.array([[0, 4], [5, 7]])
    g2_orders = np.array([3])
    g2_ijs = np.array([[5, 0], [3, 0], [2, 0], [1, 0], [6, 0]])
    g2_endpts = np.array([[0, 4]])

    (
        solved_g1_orders,
        solved_g1_ijs,
        solved_g1_endpts,
        solved_g2_orders,
        solved_g2_ijs,
        solved_g2_endpts,
    ) = solve_graph_overlaps(
        g1_orders,
        g1_ijs,
        g1_endpts,
        g2_orders,
        g2_ijs,
        g2_endpts,
        allows_arcs_overlap=True,
    )

    assert solved_g1_orders.size == 4
    assert solved_g2_orders.size == 3
    vert_vert, intr_intr, _, _ = find_graph_overlaps(
        solved_g1_ijs,
        solved_g1_endpts,
        solved_g2_ijs,
        solved_g2_endpts,
    )
    np.testing.assert_array_equal(vert_vert, np.array([[1, 0], [3, 0]]))
    np.testing.assert_array_equal(intr_intr, np.array([[2, 0]]))


def test_solve_graph_overlaps_is_idempotent():
    from formosa.geomorphology.flowdir.graphs.graphs import solve_graph_overlaps

    args = (
        np.array([1]),
        np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]]),
        np.array([[0, 4]]),
        np.array([2]),
        np.array([[1, -1], [1, 0], [2, 0], [3, 0], [3, 1]]),
        np.array([[0, 4]]),
    )

    first = solve_graph_overlaps(*args, allows_arcs_overlap=True)
    second = solve_graph_overlaps(*first, allows_arcs_overlap=True)

    for first_array, second_array in zip(first, second):
        np.testing.assert_array_equal(second_array, first_array)


@pytest.mark.parametrize("name", ("valids", "indegs"))
def test_find_acyclic_flowdirs_rejects_shape_mismatch(name):
    kwargs = {name: np.ones((2, 1), dtype=bool)}
    with pytest.raises(ValueError, match="Shapes"):
        flowdir.find_acyclic_flowdirs(
            np.zeros((1, 1), dtype=np.uint8), backend="python", **kwargs  # type: ignore
        )


if __name__ == "__main__":
    test_indegree_3x3()
    test_network_graph_3x3()
    test_network_graph_concat_3x3()
    test_locate_invalid_graph_topogtaphy()
    test_graph_insert_endpt()
    test_find_graph_overlaps()
    test_solve_graph_overlaps_with_repeated_coordinates()
    test_find_graph_overlaps_is_symmetric_with_repeated_coordinates()
    test_solve_graph_overlaps_is_idempotent()
