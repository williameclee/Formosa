"""
Tests flow-graph overlap resolution using the Fortran backend.

Last modified: 2026-08-24, En-Chi Lee (williameclee@gmail.com)
"""

from types import SimpleNamespace

import numpy as np
import pytest
from pytest import MonkeyPatch

import formosa.geomorphology.drainage.network.overlaps as ovlp_m
import formosa.geomorphology.drainage.network.validation as val_m


def test_topology_repair_simplifies_each_conflicting_arc_once(
    monkeypatch: MonkeyPatch,
):
    """
    Each conflicting arc is simplified once per repair iteration.
    """
    intersections = np.array(
        [
            [0, 1, 0, 2, 1],
            [0, 1, 0, 2, 4],
            [0, 2, 0, 4, 1],
            [1, 2, 2, 4, 1],
        ],
        dtype=np.int32,
    )
    locator_results = iter([intersections, None])
    simplified_starts = []

    def fake_locator(vtxs, endpts, backend="fortran"):
        return next(locator_results)

    def fake_simplify(vtxs, endpts, tol):
        simplified_starts.append(tuple(vtxs[:, 0]))
        return np.ones(vtxs.shape[1], dtype=np.int8)

    monkeypatch.setattr(val_m, "locate_invalid_graph_topology", fake_locator)
    monkeypatch.setattr(
        ovlp_m, "simp_f", SimpleNamespace(simplify_flowgraph=fake_simplify)
    )

    vtxs = np.array([[0, 1, 2, 3, 4, 5], [0, 0, 0, 0, 0, 0]], dtype=np.float32)
    endpts = np.array([[0, 2, 4], [1, 3, 5]], dtype=np.int32)
    keeps = ovlp_m._resolve_topology_intersections(
        vtxs, endpts, np.ones(6, dtype=bool), tol=1.0, max_iters=1
    )

    assert simplified_starts == [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)]
    np.testing.assert_array_equal(keeps, np.ones(6, dtype=bool))


def test_topology_repair_attempt_count_matches_max_iters(
    monkeypatch: MonkeyPatch,
):
    intersections = np.array([[0, 1, 0, 2, 1]], dtype=np.int32)
    simplify_calls = []

    monkeypatch.setattr(
        val_m, "locate_invalid_graph_topology", lambda *args, **kwargs: intersections
    )

    def fake_simplify(vtxs, endpts, tol):
        simplify_calls.append(tol)
        return np.ones(vtxs.shape[1], dtype=np.int8)

    monkeypatch.setattr(
        ovlp_m, "simp_f", SimpleNamespace(simplify_flowgraph=fake_simplify)
    )

    vertices = np.array([[0, 1, 2, 3], [0, 0, 0, 0]], dtype=np.float32)
    endpts = np.array([[0, 2], [1, 3]], dtype=np.int32)
    _ = ovlp_m._resolve_topology_intersections(
        vertices, endpts, np.ones(4, dtype=bool), tol=1.0, max_iters=0
    )

    assert simplify_calls == []
