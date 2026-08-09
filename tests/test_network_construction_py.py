# Last modified
#   2026-07-09, En-Chi Lee (williameclee@gmail.com)
#     - Added test case for the Python implementation of
#       `construct_flowgraph`

from tests.core import *

import pytest
import numpy as np

from formosa import D8Directions
import formosa.geomorphology.flowdir.network as nwork_m
import formosa.geomorphology.flowdir.network.construction as constr_m
import formosa.geomorphology.flowdir.network._backends.construction_py as constr_py


def test_construct_flowgraph_3x3():
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
    arc_orders, vertex_ijs, arc_endpts = constr_m.construct_flowgraph(
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
        constr_py,
        "construct_flowgraph",
        omit_selected_edge,
    )

    with pytest.raises(nwork_m.IncompleteFlowGraphError) as exc_info:
        constr_m.construct_flowgraph(
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
        constr_py,
        "construct_flowgraph",
        omit_second_edge,
    )

    with pytest.raises(nwork_m.IncompleteFlowGraphError) as exc_info:
        constr_m.construct_flowgraph(
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


def test_construct_flowgraph_can_return_adjacent_arc_ranges():
    dir_scheme = D8Directions(transform_codes=lambda x: x)
    dirs = np.array([[3, 3, 3], [3, 3, 3], [1, 1, 0]])
    valids = np.array([[T, F, T], [T, T, T], [T, T, T]])

    orders, vertices, endpts = constr_m.construct_flowgraph(
        dirs,
        dir_scheme=dir_scheme,
        backend="python",
        min_order=1,
        valids=valids,
        remove_unused=True,
    )

    assert orders.shape[0] == endpts.shape[0]
    np.testing.assert_array_equal(endpts[1:, 0], endpts[:-1, 1] + 1)
    assert vertices.shape[0] == np.sum(endpts[:, 1] - endpts[:, 0] + 1)
