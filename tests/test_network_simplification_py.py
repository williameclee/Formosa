"""
Tests flow-graph simplification using the Python backend.

Last modified: 2026-08-10, En-Chi Lee (williameclee@gmail.com)
"""

import pytest
import numpy as np

import formosa.geomorphology.drainage.network.simplification as simp_m


def test_simplify_flowgraph_can_remove_unused_vertices(monkeypatch):
    orders = np.array([1, 2], dtype=np.int8)
    vertices = np.array([[0, 0], [1, 0]])
    endpts = np.array([[0, 0], [1, 1]], dtype=np.int32)
    simplified_vertices = np.array([[0, 0], [1, 0], [99, 99], [2, 0], [3, 0]])
    simplified_endpts = np.array([[0, 1], [3, 4]], dtype=np.int32)
    keeps = np.ones(vertices.shape[0], dtype=bool)

    def fake_simplify(*args, **kwargs):
        return orders, simplified_vertices, simplified_endpts, keeps

    monkeypatch.setattr(simp_m, "_simplify_single_flowgraph", fake_simplify)
    out_orders, out_vertices, out_endpts, out_keeps = simp_m.simplify_flowgraph(
        orders, vertices, endpts, remove_unused=True
    )

    np.testing.assert_array_equal(out_orders, orders)
    np.testing.assert_array_equal(out_keeps, keeps)
    np.testing.assert_array_equal(out_vertices, [[0, 0], [1, 0], [2, 0], [3, 0]])
    np.testing.assert_array_equal(out_endpts, [[0, 1], [2, 3]])


@pytest.mark.parametrize("collection_type", [list, tuple])
def test_simplify_multiple_flowgraphs_can_remove_unused_vertices(
    monkeypatch: pytest.MonkeyPatch, collection_type
):
    orders = collection_type([np.array([1], dtype=np.int8)] * 2)
    vertices = collection_type([np.array([[0, 0], [1, 0]])] * 2)
    endpts = collection_type([np.array([[0, 1]], dtype=np.int32)] * 2)
    simplified_vertices = collection_type(
        [
            np.array([[99, 99], [0, 0], [1, 0]]),
            np.array([[2, 0], [3, 0], [98, 98]]),
        ]
    )
    simplified_endpts = collection_type(
        [np.array([[1, 2]], dtype=np.int32), np.array([[0, 1]], dtype=np.int32)]
    )
    keeps = collection_type([np.ones(2, dtype=bool)] * 2)

    def fake_simplify(*args, **kwargs):
        return orders, simplified_vertices, simplified_endpts, keeps

    monkeypatch.setattr(simp_m, "_simplify_multiple_flowgraphs", fake_simplify)
    _, out_vertices, out_endpts, _ = simp_m.simplify_flowgraph(
        orders, vertices, endpts, remove_unused=True
    )

    assert isinstance(out_vertices, collection_type)
    assert isinstance(out_endpts, collection_type)
    np.testing.assert_array_equal(out_vertices[0], [[0, 0], [1, 0]])
    np.testing.assert_array_equal(out_vertices[1], [[2, 0], [3, 0]])
    np.testing.assert_array_equal(out_endpts[0], [[0, 1]])
    np.testing.assert_array_equal(out_endpts[1], [[0, 1]])
