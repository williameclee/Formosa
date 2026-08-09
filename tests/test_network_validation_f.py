from types import SimpleNamespace

import pytest
import warnings
import numpy as np

import formosa.geomorphology.flowdir.network.validation as val_m
from formosa.geomorphology.flowdir_f import network_validation as val_f


def _make_separated_x_pairs(
    npairs: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct isolated two-segment X crossings for locator regression tests.

    Each pair contributes exactly one intersection, and the spacing between
    pairs prevents unintended intersections.
    """
    vertices = []
    endpts = []
    for ipair in range(npairs):
        x = 3 * ipair
        start = len(vertices)
        vertices.extend(
            [
                [x, 0],
                [x + 1, 1],
                [x, 1],
                [x + 1, 0],
            ]
        )
        endpts.extend([[start, start + 1], [start + 2, start + 3]])

    return (
        np.asarray(vertices, dtype=np.float32),
        np.asarray(endpts, dtype=np.int32),
    )


def test_locate_invalid_graph_topogtaphy():
    vs = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
    endpts = np.array([[0, 1], [2, 3]])
    exp_intxs = np.array([[0, 1, 0, 2, 1]], dtype=np.int32)
    intxs = val_m.locate_invalid_graph_topology(vs, endpts, backend="fortran")
    np.testing.assert_array_equal(intxs, exp_intxs)

    vs = np.array([[0, 0], [1, 1], [2, 0], [1, 0], [0, 1]])
    endpts = np.array([[0, 2], [3, 4]])
    exp_intxs = np.array([[0, 1, 0, 3, 1]], dtype=np.int32)
    intxs = val_m.locate_invalid_graph_topology(vs, endpts, backend="fortran")
    np.testing.assert_array_equal(intxs, exp_intxs)

    # Test self-intersection within a single arc
    vs = np.array([[0, 0], [2, 2], [2, 0], [0, 2]])
    endpts = np.array([[0, 3]])
    exp_intxs = np.array([[0, 0, 0, 2, 1]], dtype=np.int32)
    intxs = val_m.locate_invalid_graph_topology(vs, endpts, backend="fortran")
    np.testing.assert_array_equal(intxs, exp_intxs)

    # Test no violations
    vs = np.array([[0, 0], [1, 1], [2, 2]])
    endpts = np.array([[0, 2]])
    assert val_m.locate_invalid_graph_topology(vs, endpts, backend="fortran") is None

    # Test error handling on invalid shapes
    with pytest.raises(ValueError, match="Invalid array shapes passed"):
        val_m.locate_invalid_graph_topology(
            np.array([1, 2, 3]), endpts, backend="fortran"
        )


def test_locate_invalid_graph_topology_retries_after_buffer_overflow():
    """
    The public FORTRAN backend returns all results after provisional overflow.
    """
    vertices, endpts = _make_separated_x_pairs()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        intxs_f = val_m.locate_invalid_graph_topology(
            vertices, endpts, backend="fortran"
        )

    intxs_py = val_m.locate_invalid_graph_topology(vertices, endpts, backend="python")
    assert not caught
    assert intxs_f is not None
    assert intxs_f.shape == (5, 5)
    assert intxs_f.dtype == np.int32
    assert intxs_f.flags.c_contiguous
    np.testing.assert_array_equal(intxs_f, intxs_py)


def _scan_topology_with_capacity(
    vertices: np.ndarray,
    endpts: np.ndarray,
    capacity: int,
) -> tuple[np.ndarray, int, int]:
    """
    Call the low-level scanner after converting arrays to its FORTRAN layout.
    """
    vertices_f = np.asfortranarray(vertices.T, dtype=np.float32)
    endpts_f = np.asfortranarray(endpts.T + 1, dtype=np.int32)
    return val_f.scan_invalid_graph_topology(vertices_f, endpts_f, capacity)


def test_topology_scanner_counts_past_capacity():
    """
    The low-level scanner reports the exact count beyond storage capacity.
    """
    vertices, endpts = _make_separated_x_pairs()
    intxs, nintxs, err_code = _scan_topology_with_capacity(vertices, endpts, capacity=1)

    assert err_code == 0
    assert nintxs == 5
    assert intxs.shape == (5, 1)
    np.testing.assert_array_equal(intxs[:, 0], [1, 2, 1, 3, 1])


@pytest.mark.parametrize("capacity", [4, 5, 8])
def test_topology_scanner_capacity_boundaries(capacity):
    """
    Stored records and counts are correct around the exact capacity.
    """
    vertices, endpts = _make_separated_x_pairs()
    intxs, nintxs, err_code = _scan_topology_with_capacity(vertices, endpts, capacity)

    assert err_code == 0
    assert nintxs == 5
    assert intxs.shape == (5, capacity)

    nstored = min(nintxs, capacity)
    public_intxs = val_m.locate_invalid_graph_topology(
        vertices, endpts, backend="fortran"
    )
    assert public_intxs is not None
    expected_stored = public_intxs[:nstored].T.copy()
    expected_stored[:-1] += 1
    np.testing.assert_array_equal(intxs[:, :nstored], expected_stored)


def test_topology_scanner_empty_input_initialises_outputs():
    """
    An empty graph produces initialized scanner outputs and public `None`.
    """
    vertices = np.empty((0, 2), dtype=np.float32)
    endpts = np.empty((0, 2), dtype=np.int32)
    intxs, nintxs, err_code = _scan_topology_with_capacity(vertices, endpts, capacity=3)

    assert err_code == 0
    assert nintxs == 0
    assert intxs.shape == (5, 3)
    assert (
        val_m.locate_invalid_graph_topology(vertices, endpts, backend="fortran") is None
    )


def test_topology_wrapper_uses_single_scan_when_capacity_is_sufficient(monkeypatch):
    """
    The wrapper avoids retrying when its provisional buffer is sufficient.
    """
    calls = []

    def fake_scan(vertex_ijs, arc_endpts, capacity):
        calls.append(capacity)
        intxs = np.empty((5, capacity), dtype=np.int32, order="F")
        intxs[:, 0] = [1, 2, 1, 3, 1]
        return intxs, 1, 0

    monkeypatch.setattr(
        val_m,
        "val_f",
        SimpleNamespace(scan_invalid_graph_topology=fake_scan),
    )
    intxs = val_m.locate_invalid_graph_topology(
        np.zeros((4, 2)), np.array([[0, 1], [2, 3]]), backend="fortran"
    )

    assert calls == [3]
    np.testing.assert_array_equal(intxs, [[0, 1, 0, 2, 1]])


def test_topology_wrapper_retries_with_exact_reported_capacity(monkeypatch):
    """
    An overflow retry uses the total reported by the provisional scan.
    """
    calls = []

    def fake_scan(vertex_ijs, arc_endpts, capacity):
        calls.append(capacity)
        intxs = np.empty((5, capacity), dtype=np.int32, order="F")
        nintxs = 5
        nstored = min(capacity, nintxs)
        for i in range(nstored):
            intxs[:, i] = [2 * i + 1, 2 * i + 2, 2 * i + 1, 2 * i + 3, 1]
        return intxs, nintxs, 0

    monkeypatch.setattr(
        val_m,
        "val_f",
        SimpleNamespace(scan_invalid_graph_topology=fake_scan),
    )
    intxs = val_m.locate_invalid_graph_topology(
        np.zeros((20, 2)),
        np.arange(20, dtype=np.int32).reshape(10, 2),
        backend="fortran",
    )

    assert calls == [3, 5]
    assert intxs is not None
    assert intxs.shape == (5, 5)


def test_topology_wrapper_rejects_inconsistent_retry_count(monkeypatch):
    """
    A changed count during the deterministic retry raises an error.
    """
    calls = []

    def fake_scan(vertex_ijs, arc_endpts, capacity):
        calls.append(capacity)
        intxs = np.empty((5, capacity), dtype=np.int32, order="F")
        return intxs, 4 if len(calls) == 1 else 3, 0

    monkeypatch.setattr(
        val_m,
        "val_f",
        SimpleNamespace(scan_invalid_graph_topology=fake_scan),
    )
    with pytest.raises(RuntimeError, match="count changed"):
        val_m.locate_invalid_graph_topology(
            np.zeros((4, 2)),
            np.array([[0, 1], [2, 3]]),
            backend="fortran",
        )

    assert calls == [3, 4]


@pytest.mark.parametrize(
    ("err_code", "exception"),
    [(1, ValueError), (2, MemoryError), (99, RuntimeError)],
)
def test_topology_wrapper_translates_scanner_errors(monkeypatch, err_code, exception):
    """
    Scanner status codes map to the documented Python exceptions.
    """

    def fake_scan(vertex_ijs, arc_endpts, capacity):
        return np.empty((5, capacity), dtype=np.int32), 0, err_code

    monkeypatch.setattr(
        val_m,
        "val_f",
        SimpleNamespace(scan_invalid_graph_topology=fake_scan),
    )
    with pytest.raises(exception):
        val_m.locate_invalid_graph_topology(
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
