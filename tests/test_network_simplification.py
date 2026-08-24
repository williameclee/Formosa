"""
Tests backend-independent flow-graph simplification wrappers.

Last modified: 2026-08-24, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np
import pytest
from pytest import MonkeyPatch

import formosa.geomorphology.drainage.network.simplification as simp_m


def test_simplify_flowgraph_can_remove_unused_vertices(monkeypatch: MonkeyPatch):
    orders = np.array([1, 2], dtype=np.int8)
    vtxs = np.array([[0, 0], [1, 0]])
    endpts = np.array([[0, 0], [1, 1]], dtype=np.int32)
    simp_vtxs = np.array([[0, 0], [1, 0], [99, 99], [2, 0], [3, 0]])
    simp_endpts = np.array([[0, 1], [3, 4]], dtype=np.int32)
    keeps = np.ones(vtxs.shape[0], dtype=bool)

    def fake_simplify(*args, **kwargs):
        return orders, simp_vtxs, simp_endpts, keeps

    monkeypatch.setattr(simp_m, "_simplify_single_flowgraph", fake_simplify)
    out_orders, out_vtxs, out_endpts, out_keeps = simp_m.simplify_flowgraph(
        orders, vtxs, endpts, remove_unused=True
    )

    np.testing.assert_array_equal(out_orders, orders)
    np.testing.assert_array_equal(out_keeps, keeps)
    np.testing.assert_array_equal(out_vtxs, [[0, 0], [1, 0], [2, 0], [3, 0]])
    np.testing.assert_array_equal(out_endpts, [[0, 1], [2, 3]])


@pytest.mark.parametrize("collection_type", [list, tuple])
def test_simplify_multiple_flowgraphs_can_remove_unused_vertices(
    monkeypatch: pytest.MonkeyPatch, collection_type
):
    orders = collection_type([np.array([1], dtype=np.int8)] * 2)
    vtxs = collection_type([np.array([[0, 0], [1, 0]])] * 2)
    endpts = collection_type([np.array([[0, 1]], dtype=np.int32)] * 2)
    simp_vtxs = collection_type(
        [
            np.array([[99, 99], [0, 0], [1, 0]]),
            np.array([[2, 0], [3, 0], [98, 98]]),
        ]
    )
    simp_endpts = collection_type(
        [np.array([[1, 2]], dtype=np.int32), np.array([[0, 1]], dtype=np.int32)]
    )
    keeps = collection_type([np.ones(2, dtype=bool)] * 2)

    def fake_simplify(*args, **kwargs):
        return orders, simp_vtxs, simp_endpts, keeps

    monkeypatch.setattr(simp_m, "_simplify_multiple_flowgraphs", fake_simplify)
    _, out_vtxs, out_endpts, _ = simp_m.simplify_flowgraph(
        orders, vtxs, endpts, remove_unused=True
    )

    assert isinstance(out_vtxs, collection_type)
    assert isinstance(out_endpts, collection_type)
    np.testing.assert_array_equal(out_vtxs[0], [[0, 0], [1, 0]])
    np.testing.assert_array_equal(out_vtxs[1], [[2, 0], [3, 0]])
    np.testing.assert_array_equal(out_endpts[0], [[0, 1]])
    np.testing.assert_array_equal(out_endpts[1], [[0, 1]])
