"""
Tests flow-graph topology validation using the Fortran backend.

Last modified: 2026-08-24, En-Chi Lee (williameclee@gmail.com)
"""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from formosa.geomorphology._native import network_validation as val_f
from pytest import MonkeyPatch

import formosa.geomorphology.drainage.network.validation as val_m


def _make_separated_x_pairs(
    npairs: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct isolated two-segment X crossings for locator regression tests.

    Each pair contributes exactly one intersection, and the spacing between
    pairs prevents unintended intersections.
    """
    vtxs = []
    endpts = []
    for ipair in range(npairs):
        x = 3 * ipair
        start = len(vtxs)
        vtxs.extend([[x, 0], [x + 1, 1], [x, 1], [x + 1, 0]])
        endpts.extend([[start, start + 1], [start + 2, start + 3]])

    return (
        np.asarray(vtxs, dtype=np.float32),
        np.asarray(endpts, dtype=np.int32),
    )


def test_locate_invalid_graph_topology_retries_after_buffer_overflow():
    """
    The public Fortran backend returns all results after provisional overflow.
    """
    vtxs, endpts = _make_separated_x_pairs()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        intxs_f = val_m.locate_invalid_graph_topology(vtxs, endpts, backend="fortran")

    intxs_py = val_m.locate_invalid_graph_topology(vtxs, endpts, backend="python")
    assert not caught
    assert intxs_f is not None
    assert intxs_f.shape == (5, 5)
    assert intxs_f.dtype == np.int32
    assert intxs_f.flags.c_contiguous
    np.testing.assert_array_equal(intxs_f, intxs_py)


def _scan_topology_with_capacity(
    vtxs: np.ndarray, endpts: np.ndarray, cpty: int
) -> tuple[np.ndarray, int, int]:
    """
    Call the low-level scanner after converting arrays to its Fortran layout.
    """
    vertices_f = np.asfortranarray(vtxs.T, dtype=np.float32)
    endpts_f = np.asfortranarray(endpts.T + 1, dtype=np.int32)
    return val_f.scan_invalid_graph_topology(vertices_f, endpts_f, cpty)


def test_topology_scanner_counts_past_capacity():
    """
    The low-level scanner reports the exact count beyond storage capacity.
    """
    vertices, endpts = _make_separated_x_pairs()
    intxs, nintxs, err_code = _scan_topology_with_capacity(vertices, endpts, cpty=1)

    assert err_code == 0
    assert nintxs == 5
    assert intxs.shape == (5, 1)
    np.testing.assert_array_equal(intxs[:, 0], [1, 2, 1, 3, 1])


@pytest.mark.parametrize("cpty", [4, 5, 8])
def test_topology_scanner_capacity_boundaries(cpty: int):
    """
    Stored records and counts are correct around the exact capacity.
    """
    vtxs, endpts = _make_separated_x_pairs()
    intxs, nintxs, err_code = _scan_topology_with_capacity(vtxs, endpts, cpty)

    assert err_code == 0
    assert nintxs == 5
    assert intxs.shape == (5, cpty)

    nstored = min(nintxs, cpty)
    public_intxs = val_m.locate_invalid_graph_topology(vtxs, endpts, backend="fortran")
    assert public_intxs is not None
    exp_stored = public_intxs[:nstored].T.copy()
    exp_stored[:-1] += 1
    np.testing.assert_array_equal(intxs[:, :nstored], exp_stored)


def test_topology_scanner_empty_input_initialises_outputs():
    """
    An empty graph produces initialized scanner outputs and public `None`.
    """
    vtxs = np.empty((0, 2), dtype=np.float32)
    endpts = np.empty((0, 2), dtype=np.int32)
    intxs, nintxs, err_code = _scan_topology_with_capacity(vtxs, endpts, cpty=3)

    assert err_code == 0
    assert nintxs == 0
    assert intxs.shape == (5, 3)
    assert val_m.locate_invalid_graph_topology(vtxs, endpts, backend="fortran") is None


def test_topology_wrapper_uses_single_scan_when_capacity_is_sufficient(
    monkeypatch: pytest.MonkeyPatch,
):
    """
    The wrapper avoids retrying when its provisional buffer is sufficient.
    """
    calls = []

    def fake_scan(vtxs, endpts, cpty):
        calls.append(cpty)
        intxs = np.empty((5, cpty), dtype=np.int32, order="F")
        intxs[:, 0] = [1, 2, 1, 3, 1]
        return intxs, 1, 0

    monkeypatch.setattr(
        val_m, "val_f", SimpleNamespace(scan_invalid_graph_topology=fake_scan)
    )
    intxs = val_m.locate_invalid_graph_topology(
        np.zeros((4, 2)), np.array([[0, 1], [2, 3]]), backend="fortran"
    )

    assert calls == [3]
    np.testing.assert_array_equal(intxs, [[0, 1, 0, 2, 1]])


def test_topology_wrapper_retries_with_exact_reported_capacity(
    monkeypatch: pytest.MonkeyPatch,
):
    """
    An overflow retry uses the total reported by the provisional scan.
    """
    calls = []

    def fake_scan(vtxs, endpts, cpty):
        calls.append(cpty)
        intxs = np.empty((5, cpty), dtype=np.int32, order="F")
        nintxs = 5
        nstored = min(cpty, nintxs)
        for i in range(nstored):
            intxs[:, i] = [2 * i + 1, 2 * i + 2, 2 * i + 1, 2 * i + 3, 1]
        return intxs, nintxs, 0

    monkeypatch.setattr(
        val_m, "val_f", SimpleNamespace(scan_invalid_graph_topology=fake_scan)
    )
    intxs = val_m.locate_invalid_graph_topology(
        np.zeros((20, 2)),
        np.arange(20, dtype=np.int32).reshape(10, 2),
        backend="fortran",
    )

    assert calls == [3, 5]
    assert intxs is not None
    assert intxs.shape == (5, 5)


def test_topology_wrapper_rejects_inconsistent_retry_count(
    monkeypatch: pytest.MonkeyPatch,
):
    """
    A changed count during the deterministic retry raises an error.
    """
    calls = []

    def fake_scan(vtxs, endpts, cpty):
        calls.append(cpty)
        intxs = np.empty((5, cpty), dtype=np.int32, order="F")
        return intxs, 4 if len(calls) == 1 else 3, 0

    monkeypatch.setattr(
        val_m, "val_f", SimpleNamespace(scan_invalid_graph_topology=fake_scan)
    )
    with pytest.raises(RuntimeError, match="count changed"):
        _ = val_m.locate_invalid_graph_topology(
            np.zeros((4, 2)),
            np.array([[0, 1], [2, 3]]),
            backend="fortran",
        )

    assert calls == [3, 4]


@pytest.mark.parametrize(
    ("err_code", "exception"),
    [(1, ValueError), (2, MemoryError), (99, RuntimeError)],
)
def test_topology_wrapper_translates_scanner_errors(
    monkeypatch: MonkeyPatch, err_code: int, exception: Exception
):
    """
    Scanner status codes map to the documented Python exceptions.
    """

    def fake_scan(vtxs, endpts, cpty):
        return np.empty((5, cpty), dtype=np.int32), 0, err_code

    monkeypatch.setattr(
        val_m, "val_f", SimpleNamespace(scan_invalid_graph_topology=fake_scan)
    )
    with pytest.raises(exception):  # pyright: ignore[reportArgumentType]
        _ = val_m.locate_invalid_graph_topology(
            np.zeros((2, 2)), np.array([[0, 1]]), backend="fortran"
        )


def test_self_intersection_overflow_does_not_hide_later_interarc_results():
    """
    Self-scan overflow does not hide later inter-arc violations.
    """
    vertices = np.array(
        [
            [0, 0],
            [1, 1],
            [0, 0],
            [1, 1],
            [0, 0],
            [1, 1],
            [0, 0],
            [10, 0],
            [11, 1],
            [10, 1],
            [11, 0],
        ],
        dtype=np.float32,
    )
    endpts = np.array([[0, 6], [7, 8], [9, 10]], dtype=np.int32)

    intxs_f = val_m.locate_invalid_graph_topology(vertices, endpts, backend="fortran")
    intxs_py = val_m.locate_invalid_graph_topology(vertices, endpts, backend="python")

    assert intxs_f is not None
    assert intxs_f.shape[0] > 3
    assert np.any((intxs_f[:, 0] == 0) & (intxs_f[:, 1] == 0))
    assert np.any((intxs_f[:, 0] == 1) & (intxs_f[:, 1] == 2))
    np.testing.assert_array_equal(intxs_f, intxs_py)


def test_large_self_intersection_only_result_matches_python_backend():
    """
    A dense self-intersection result stays complete across backends.
    """
    vertices = np.array(
        [[0, 0], [1, 1], [0, 0], [1, 1], [0, 0], [1, 1], [0, 0]],
        dtype=np.float32,
    )
    endpts = np.array([[0, 6]], dtype=np.int32)

    intxs_f = val_m.locate_invalid_graph_topology(vertices, endpts, backend="fortran")
    intxs_py = val_m.locate_invalid_graph_topology(vertices, endpts, backend="python")

    assert intxs_f is not None
    assert intxs_f.shape[0] > 3
    assert np.all(intxs_f[:, :2] == 0)
    np.testing.assert_array_equal(intxs_f, intxs_py)


def test_topology_results_are_invariant_to_arc_reordering():
    """
    Arc reordering preserves the geometric intersection set.
    """
    vertices, endpts = _make_separated_x_pairs()
    baseline = val_m.locate_invalid_graph_topology(vertices, endpts, backend="fortran")

    permutation = np.arange(endpts.shape[0] - 1, -1, -1)
    reordered = val_m.locate_invalid_graph_topology(
        vertices, endpts[permutation], backend="fortran"
    )

    def remap_rows(rows, arc_ids):
        remapped = set()
        for iarc, jarc, iseg, jseg, intx_flag in rows:
            iarc = int(arc_ids[iarc])
            jarc = int(arc_ids[jarc])
            if iarc > jarc:
                iarc, jarc = jarc, iarc
                iseg, jseg = jseg, iseg
            remapped.add((iarc, jarc, int(iseg), int(jseg), int(intx_flag)))
        return remapped

    assert remap_rows(baseline, np.arange(endpts.shape[0])) == remap_rows(
        reordered, permutation
    )
