"""
Tests related to the resolution of invalid graph topologies using 
the Python backend.

Last modified: 2026-08-09, En-Chi Lee (williameclee@gmail.com)
"""

import pytest
import numpy as np

import formosa.geomorphology.drainage.network.overlaps as ovlp_m


def test_solve_graph_overlaps_can_remove_unused_vertices():
    orders = np.array([1], dtype=np.int8)
    g1_vertices = np.array([[99, 99], [0, 0], [1, 0], [98, 98]])
    g2_vertices = np.array([[97, 97], [0, 1], [1, 1], [96, 96]])
    endpts = np.array([[1, 2]], dtype=np.int32)

    _, out_g1_vertices, out_g1_endpts, _, out_g2_vertices, out_g2_endpts = (
        ovlp_m.solve_graph_overlaps(
            orders, g1_vertices, endpts, orders, g2_vertices, endpts, remove_unused=True
        )
    )
    np.testing.assert_array_equal(out_g1_vertices, [[0, 0], [1, 0]])
    np.testing.assert_array_equal(out_g2_vertices, [[0, 1], [1, 1]])
    np.testing.assert_array_equal(out_g1_endpts, [[0, 1]])
    np.testing.assert_array_equal(out_g2_endpts, [[0, 1]])


def test_find_graph_overlaps():
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

    vert_vert, intr_intr, g1_intr_g2_vert, g1_vert_g2_intr = ovlp_m.find_graph_overlaps(
        g1_ijs, g1_endpts, g2_ijs, g2_endpts
    )

    np.testing.assert_array_equal(vert_vert, np.array([[0, 0]]))
    np.testing.assert_array_equal(intr_intr, np.array([[4, 0]]))
    np.testing.assert_array_equal(g1_intr_g2_vert, np.array([[1, 0]]))
    np.testing.assert_array_equal(g1_vert_g2_intr, np.array([[2, 0]]))

    # The matching unused coordinate must not be reported as an overlap
    for overlap_type in (vert_vert, intr_intr, g1_intr_g2_vert, g1_vert_g2_intr):
        assert not np.any(np.all(overlap_type == [99, 99], axis=1))


def test_find_graph_overlaps_is_symmetric_with_repeated_coordinates():
    g1_ijs = np.array([[0, 0], [1, 0], [2, 0], [1, 0], [3, 0]])
    g1_endpts = np.array([[0, 2], [3, 4]])
    g2_ijs = np.array([[0, 0], [2, 0], [4, 0], [1, 0], [5, 0]])
    g2_endpts = np.array([[0, 2], [3, 4]])

    forward = ovlp_m.find_graph_overlaps(g1_ijs, g1_endpts, g2_ijs, g2_endpts)
    forward = ovlp_m.find_graph_overlaps(g1_ijs, g1_endpts, g2_ijs, g2_endpts)
    reverse = ovlp_m.find_graph_overlaps(g2_ijs, g2_endpts, g1_ijs, g1_endpts)

    np.testing.assert_array_equal(forward[0], reverse[0])
    np.testing.assert_array_equal(forward[1], reverse[1])
    np.testing.assert_array_equal(forward[2], reverse[3])
    np.testing.assert_array_equal(forward[3], reverse[2])
    # A coordinate is classified as an endpoint when any used occurrence is
    # an endpoint, even if another occurrence is interior.
    assert np.any(np.all(forward[0] == [1, 0], axis=1))


@pytest.mark.parametrize("empty_graph", (1, 2))
def test_find_graph_overlaps_accepts_an_empty_graph(empty_graph):
    empty_ijs = np.empty((0, 2), dtype=np.int32)
    empty_endpts = np.empty((0, 2), dtype=np.int32)
    ijs = np.array([[0, 0], [1, 0]], dtype=np.int32)
    endpts = np.array([[0, 1]], dtype=np.int32)
    graphs = ((empty_ijs, empty_endpts), (ijs, endpts))
    if empty_graph == 2:
        graphs = graphs[::-1]

    overlaps = ovlp_m.find_graph_overlaps(*graphs[0], *graphs[1])

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
        (False, 5, np.array([[1, 0], [2, 0], [3, 0]]), np.empty((0, 2), dtype=int)),
    ],
)
def test_solve_graph_overlaps(
    allows_arcs_overlap,
    expected_narcs,
    expected_vert_vert,
    expected_intr_intr,
):
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
    ) = ovlp_m.solve_graph_overlaps(
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

    vert_vert, intr_intr, g1_intr_g2_vert, g1_vert_g2_intr = ovlp_m.find_graph_overlaps(
        solved_g1_ijs, solved_g1_endpts, solved_g2_ijs, solved_g2_endpts
    )
    np.testing.assert_array_equal(vert_vert, expected_vert_vert)
    np.testing.assert_array_equal(intr_intr, expected_intr_intr)
    assert g1_intr_g2_vert.size == 0
    assert g1_vert_g2_intr.size == 0


def test_solve_graph_overlaps_with_repeated_coordinates():
    # Coordinate (2, 0) occurs in two arcs of the first graph
    g1_orders = np.array([1, 2])
    g1_ijs = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [8, 0], [2, 0], [9, 0]])
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
    ) = ovlp_m.solve_graph_overlaps(
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
    vert_vert, intr_intr, _, _ = ovlp_m.find_graph_overlaps(
        solved_g1_ijs,
        solved_g1_endpts,
        solved_g2_ijs,
        solved_g2_endpts,
    )
    np.testing.assert_array_equal(vert_vert, np.array([[1, 0], [3, 0]]))
    np.testing.assert_array_equal(intr_intr, np.array([[2, 0]]))


def test_solve_graph_overlaps_is_idempotent():
    args = (
        np.array([1]),
        np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]]),
        np.array([[0, 4]]),
        np.array([2]),
        np.array([[1, -1], [1, 0], [2, 0], [3, 0], [3, 1]]),
        np.array([[0, 4]]),
    )

    first = ovlp_m.solve_graph_overlaps(*args, allows_arcs_overlap=True)
    second = ovlp_m.solve_graph_overlaps(*first, allows_arcs_overlap=True)

    for first_array, second_array in zip(first, second):
        np.testing.assert_array_equal(second_array, first_array)
