# Last modified
#   2026-07-09, En-Chi Lee (williameclee@gmail.com)
#     - Added test case for the Fortran implementation of
#       `construct_flowgraph`.
#   2026-07-12, En-Chi Lee (williameclee@gmail.com)
#     - Added test cases for function `test_locate_invalid_graph_topogtaphy`.
#   2026-07-29, En-Chi Lee (williameclee@gmail.com)
#     - Added test cases for function `simplify_flowgraph`.
#     - Added complete topology-intersection scan-and-retry
#       regression tests.
#   2026-07-31, En-Chi Lee (williameclee@gmail.com)
#     - Updated tests to match the updated `simplify_flowgraph`
#       interface; also added additional tests for the new interface.

import pytest
import warnings
import numpy as np

from formosa import D8Directions
import formosa.geomorphology.flowdir.network as nwork_m
from formosa.geomorphology.flowdir.network import graphs as graphs_m
from formosa.geomorphology.flowdir_f import flowdir_graphs as graphs_f

from types import SimpleNamespace

T = True
F = False


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
    arc_orders, vertex_ijs, arc_endpts = graphs_m.construct_flowgraph(
        dirs, dir_scheme=dir_scheme, backend="fortran", min_order=1, valids=valids
    )
    arc_lengths = arc_endpts[:, 1] - arc_endpts[:, 0]

    np.testing.assert_array_equal(arc_orders, exp_orders)
    np.testing.assert_array_equal(arc_lengths, exp_lengths)

    for i, exp_ij in enumerate(exp_ijs):
        np.testing.assert_array_equal(
            vertex_ijs[arc_endpts[i, 0] : arc_endpts[i, 1] + 1], exp_ij
        )


def test_construct_flowgraph_fortran_returns_buffer_overflow_code():
    dirs = np.array([[1, 1, 0]], dtype=np.uint8, order="F")
    valids = np.ones((1, 3), dtype=bool, order="F")
    orders = np.ones((1, 3), dtype=np.int16, order="F")
    seeds = np.array([[True, False, False]], dtype=bool, order="F")
    indegs = np.array([[0, 1, 1]], dtype=np.int8, order="F")
    offsets = np.array([[0, 1], [0, 0]], dtype=np.int32, order="F")
    codes = np.array([1, 0], dtype=np.uint8, order="F")

    *_, err_code = graphs_f.construct_flowgraph(
        dirs,
        valids,
        orders,
        seeds,
        indegs,
        offsets,
        codes,
        True,
        1,
    )

    assert err_code == 3


def test_construct_flowgraph_translates_fortran_error(monkeypatch):
    def fake_construct(*args):
        return (
            0,
            0,
            np.zeros(1, dtype=np.int16),
            np.zeros((2, 2), dtype=np.int32),
            np.zeros((2, 1), dtype=np.int32),
            3,
        )

    monkeypatch.setattr(
        graphs_m,
        "graphs_f",
        SimpleNamespace(construct_flowgraph=fake_construct),
    )

    with pytest.raises(RuntimeError, match=r"construct_flowgraph.*error code 3"):
        graphs_m.construct_flowgraph(
            np.array([[0]], dtype=np.uint8),
            orders=np.ones((1, 1), dtype=np.uint8),
            min_order=1,
            backend="fortran",
        )


def test_locate_invalid_graph_topogtaphy():
    vs = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
    endpts = np.array([[0, 1], [2, 3]])
    exp_intxs = np.array([[0, 1, 0, 2, 1]], dtype=np.int32)
    intxs = nwork_m.validation.locate_invalid_graph_topology(
        vs, endpts, backend="fortran"
    )
    np.testing.assert_array_equal(intxs, exp_intxs)

    vs = np.array([[0, 0], [1, 1], [2, 0], [1, 0], [0, 1]])
    endpts = np.array([[0, 2], [3, 4]])
    exp_intxs = np.array([[0, 1, 0, 3, 1]], dtype=np.int32)
    intxs = nwork_m.validation.locate_invalid_graph_topology(
        vs, endpts, backend="fortran"
    )
    np.testing.assert_array_equal(intxs, exp_intxs)

    # Test self-intersection within a single arc
    vs = np.array([[0, 0], [2, 2], [2, 0], [0, 2]])
    endpts = np.array([[0, 3]])
    exp_intxs = np.array([[0, 0, 0, 2, 1]], dtype=np.int32)
    intxs = nwork_m.validation.locate_invalid_graph_topology(
        vs, endpts, backend="fortran"
    )
    np.testing.assert_array_equal(intxs, exp_intxs)

    # Test no violations
    vs = np.array([[0, 0], [1, 1], [2, 2]])
    endpts = np.array([[0, 2]])
    assert (
        nwork_m.validation.locate_invalid_graph_topology(vs, endpts, backend="fortran")
        is None
    )

    # Test error handling on invalid shapes
    with pytest.raises(ValueError, match="Invalid array shapes passed"):
        nwork_m.validation.locate_invalid_graph_topology(
            np.array([1, 2, 3]), endpts, backend="fortran"
        )


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
    return graphs_f.scan_invalid_graph_topology(vertices_f, endpts_f, capacity)


def test_locate_invalid_graph_topology_retries_after_buffer_overflow():
    """
    The public FORTRAN backend returns all results after provisional overflow.
    """
    vertices, endpts = _make_separated_x_pairs()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        intxs_f = nwork_m.validation.locate_invalid_graph_topology(
            vertices, endpts, backend="fortran"
        )

    intxs_py = nwork_m.validation.locate_invalid_graph_topology(
        vertices, endpts, backend="python"
    )
    assert not caught
    assert intxs_f is not None
    assert intxs_f.shape == (5, 5)
    assert intxs_f.dtype == np.int32
    assert intxs_f.flags.c_contiguous
    np.testing.assert_array_equal(intxs_f, intxs_py)


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
    public_intxs = nwork_m.validation.locate_invalid_graph_topology(
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
        nwork_m.validation.locate_invalid_graph_topology(
            vertices, endpts, backend="fortran"
        )
        is None
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
        nwork_m.validation,
        "graphs_f",
        SimpleNamespace(scan_invalid_graph_topology=fake_scan),
    )
    intxs = nwork_m.validation.locate_invalid_graph_topology(
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
        nwork_m.validation,
        "graphs_f",
        SimpleNamespace(scan_invalid_graph_topology=fake_scan),
    )
    intxs = nwork_m.validation.locate_invalid_graph_topology(
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
        nwork_m.validation,
        "graphs_f",
        SimpleNamespace(scan_invalid_graph_topology=fake_scan),
    )
    with pytest.raises(RuntimeError, match="count changed"):
        nwork_m.validation.locate_invalid_graph_topology(
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
        nwork_m.validation,
        "graphs_f",
        SimpleNamespace(scan_invalid_graph_topology=fake_scan),
    )
    with pytest.raises(exception):
        nwork_m.validation.locate_invalid_graph_topology(
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

    intxs_f = nwork_m.validation.locate_invalid_graph_topology(
        vertices, endpts, backend="fortran"
    )
    intxs_py = nwork_m.validation.locate_invalid_graph_topology(
        vertices, endpts, backend="python"
    )

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

    intxs_f = nwork_m.validation.locate_invalid_graph_topology(
        vertices, endpts, backend="fortran"
    )
    intxs_py = nwork_m.validation.locate_invalid_graph_topology(
        vertices, endpts, backend="python"
    )

    assert intxs_f is not None
    assert intxs_f.shape[0] > 3
    assert np.all(intxs_f[:, :2] == 0)
    np.testing.assert_array_equal(intxs_f, intxs_py)


def test_topology_results_are_invariant_to_arc_reordering():
    """
    Arc reordering preserves the geometric intersection set.
    """
    vertices, endpts = _make_separated_x_pairs()
    baseline = nwork_m.validation.locate_invalid_graph_topology(
        vertices, endpts, backend="fortran"
    )

    permutation = np.arange(endpts.shape[0] - 1, -1, -1)
    reordered = nwork_m.validation.locate_invalid_graph_topology(
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


def test_topology_repair_simplifies_each_conflicting_arc_once(monkeypatch):
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

    def fake_locator(vertex_xys, arc_endpts, backend="fortran"):
        return next(locator_results)

    def fake_simplify(vertex_xys, arc_endpts, tol):
        simplified_starts.append(tuple(vertex_xys[:, 0]))
        return np.ones(vertex_xys.shape[1], dtype=np.int8)

    monkeypatch.setattr(graphs_m, "locate_invalid_graph_topology", fake_locator)
    monkeypatch.setattr(
        graphs_m,
        "graphs_f",
        SimpleNamespace(simplify_flowgraph=fake_simplify),
    )

    vertices = np.array([[0, 1, 2, 3, 4, 5], [0, 0, 0, 0, 0, 0]], dtype=np.float32)
    endpts = np.array([[0, 2, 4], [1, 3, 5]], dtype=np.int32)
    keeps = graphs_m._resolve_topology_intersections(
        vertices, endpts, np.ones(6, dtype=bool), tol=1.0, max_iters=1
    )

    assert simplified_starts == [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)]
    np.testing.assert_array_equal(keeps, np.ones(6, dtype=bool))


def test_topology_repair_attempt_count_matches_max_iters(monkeypatch):
    intersections = np.array([[0, 1, 0, 2, 1]], dtype=np.int32)
    simplify_calls = []

    monkeypatch.setattr(
        graphs_m,
        "locate_invalid_graph_topology",
        lambda *args, **kwargs: intersections,
    )

    def fake_simplify(vertex_xys, arc_endpts, tol):
        simplify_calls.append(tol)
        return np.ones(vertex_xys.shape[1], dtype=np.int8)

    monkeypatch.setattr(
        graphs_m,
        "graphs_f",
        SimpleNamespace(simplify_flowgraph=fake_simplify),
    )

    vertices = np.array([[0, 1, 2, 3], [0, 0, 0, 0]], dtype=np.float32)
    endpts = np.array([[0, 2], [1, 3]], dtype=np.int32)
    graphs_m._resolve_topology_intersections(
        vertices,
        endpts,
        np.ones(4, dtype=bool),
        tol=1.0,
        max_iters=0,
    )

    assert simplify_calls == []


def test_simplify_single_flowgraph():
    verts = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    endpts = np.array([[0, 2]])
    orders = np.array([2, 4])
    simp_orders, simp_verts, simp_endpts, keeps = graphs_m.simplify_flowgraph(
        np.array([3]), verts, endpts, tol=1.0, check_topology=False, backend="fortran"
    )
    np.testing.assert_array_equal(simp_orders, [3])
    np.testing.assert_array_equal(keeps, [True, False, True])
    np.testing.assert_array_equal(simp_verts, [[0.0, 0.0], [2.0, 2.0]])
    np.testing.assert_array_equal(simp_endpts, [[0, 1]])

    # Test topology correction
    verts_topo = np.array([[0.0, 0.8], [1.0, 2.0], [2.0, 0.2], [0.5, 0.5], [1.5, 0.5]])
    endpts_topo = np.array([[0, 2], [3, 4]])

    # Under tol = 1.5 and check_topology = False, simplification occurs and causes intersection
    _, _, _, keeps_no_check = graphs_m.simplify_flowgraph(
        *(orders, verts_topo, endpts_topo),
        tol=1.5,
        check_topology=False,
        backend="fortran",
    )
    # Vertex 1 should be removed
    np.testing.assert_array_equal(keeps_no_check, [True, False, True, True, True])

    # Under tol = 1.5 and check_topology = True, it detects intersection, reduces tolerance,
    # and keeps Vertex 1 to avoid intersection
    _, checked_verts, checked_endpts, keeps_with_check = graphs_m.simplify_flowgraph(
        *(orders, verts_topo, endpts_topo),
        tol=1.5,
        check_topology=True,
        backend="fortran",
    )
    # Vertex 1 should be kept
    np.testing.assert_array_equal(keeps_with_check, [True, True, True, True, True])
    assert (
        nwork_m.validation.locate_invalid_graph_topology(
            checked_verts, checked_endpts, backend="fortran"
        )
        is None
    )

    with warnings.catch_warnings():
        warnings.simplefilter("default")
        filters_before = list(warnings.filters)
        graphs_m.simplify_flowgraph(
            *(orders, verts_topo, endpts_topo),
            tol=1.5,
            check_topology=True,
            backend="fortran",
        )
        assert warnings.filters == filters_before


def test_simplify_single_flowgraph_preserves_vertex_layout():
    verts = np.array(
        [
            [0.0, 1.0],
            [1.0, 3.0],
            [2.0, 1.0],
            [3.0, 4.0],
        ]
    )
    endpts = np.array([[0, 3]])
    orders = np.array([5])

    simp_orders, simp_verts, simp_endpts, keeps = graphs_m.simplify_flowgraph(
        *(orders, verts, endpts),
        tol=0.0,
        check_topology=False,
        backend="fortran",
    )

    assert simp_verts.shape == (4, 2)
    np.testing.assert_array_equal(simp_orders, orders)
    np.testing.assert_array_equal(simp_verts, verts)
    np.testing.assert_array_equal(simp_endpts, endpts)
    np.testing.assert_array_equal(keeps, np.ones(4, dtype=bool))


def test_simplify_flowgraph_validates_arc_orders():
    verts = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    endpts = np.array([[0, 2]])
    orders = [np.array([1]), np.array([2])]

    with pytest.raises(TypeError, match="must be NumPy arrays"):
        graphs_m.simplify_flowgraph(
            np.array([1]),
            "not-an-array",  # type: ignore
            endpts,
            check_topology=False,
        )

    with pytest.raises(ValueError, match="Order array has length 0"):
        graphs_m.simplify_flowgraph(
            np.array([], dtype=np.uint8),
            verts,
            endpts,
            check_topology=False,
        )

    with pytest.raises(ValueError, match="must have the same length"):
        graphs_m.simplify_flowgraph(
            *(orders, [verts], [endpts]),
            check_topology=False,
        )


def test_simplify_rejects_invalid_final_graph_from_valid_input(monkeypatch):
    verts = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [0.5, 0.5], [1.5, 0.5]])
    endpts = np.array([[0, 2], [3, 4]])
    orders = np.array([1, 2])
    final_intxs = np.array([[0, 1, 0, 2, 1]], dtype=np.int32)

    monkeypatch.setattr(
        graphs_m,
        "_resolve_topology_intersections",
        lambda verts, endpts, keeps, tol, graph_ids=None: np.array([T, F, T, T, T]),
    )
    locator_results = iter([final_intxs, None])
    monkeypatch.setattr(
        graphs_m,
        "_locate_disallowed_graph_topology",
        lambda verts, endpts, graph_ids=None: next(locator_results),
    )

    with pytest.raises(nwork_m.validation.UnresolvedSimplificationTopology) as exc_info:
        graphs_m.simplify_flowgraph(
            *(orders, verts, endpts),
            tol=1.0,
            check_topology=True,
            backend="fortran",
        )


def test_simplify_rejects_invalid_final_graph_from_invalid_input(monkeypatch):
    verts = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    endpts = np.array([[0, 1], [2, 3]])
    orders = np.array([1, 2])
    final_intxs = np.array([[0, 1, 0, 2, 1]], dtype=np.int32)
    input_intxs = np.array([[0, 1, 0, 2, 1]], dtype=np.int32)

    monkeypatch.setattr(
        graphs_m,
        "_resolve_topology_intersections",
        lambda verts, endpts, keeps, tol, graph_ids=None: np.ones(
            verts.shape[1], dtype=bool
        ),
    )
    locator_results = iter([final_intxs, input_intxs])
    monkeypatch.setattr(
        graphs_m,
        "_locate_disallowed_graph_topology",
        lambda verts, endpts, graph_ids=None: next(locator_results),
    )

    with pytest.raises(nwork_m.validation.InvalidOriginalGraphTopology) as exc_info:
        graphs_m.simplify_flowgraph(
            *(orders, verts, endpts),
            tol=1.0,
            check_topology=True,
            backend="fortran",
        )


def test_simplify_skips_final_validation_when_topology_check_is_disabled(
    monkeypatch,
):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("Topology validation should be disabled.")

    monkeypatch.setattr(graphs_m, "_locate_disallowed_graph_topology", fail_if_called)

    graphs_m.simplify_flowgraph(
        np.array([1]),
        np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
        np.array([[0, 2]]),
        tol=1.0,
        check_topology=False,
        backend="fortran",
    )


def test_simplify_multiple_flowgraphs():
    # Llist of standard arrays
    vs0 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    endpts0 = np.array([[0, 2]], dtype=np.int32)
    orders0 = np.array([2], dtype=np.uint8)
    verts1 = np.array([[3.0, 3.0], [4.0, 4.0], [5.0, 5.0]], dtype=np.float32)
    endpts1 = np.array([[0, 2]], dtype=np.int32)
    orders1 = np.array([4], dtype=np.uint8)

    simp_orders_list, simp_verts_list, simp_endpts_list, keeps_list = (
        graphs_m.simplify_flowgraph(
            [orders0, orders1],
            [vs0, verts1],
            [endpts0, endpts1],
            tol=1.0,
            check_topology=False,
            backend="fortran",
        )
    )
    assert isinstance(simp_orders_list, list)
    assert isinstance(simp_verts_list, list)
    np.testing.assert_array_equal(simp_orders_list[0], orders0)
    np.testing.assert_array_equal(simp_orders_list[1], orders1)
    np.testing.assert_array_equal(keeps_list[0], [T, F, T])
    np.testing.assert_array_equal(keeps_list[1], [T, F, T])
    np.testing.assert_array_equal(simp_verts_list[0], [[0.0, 0.0], [2.0, 2.0]])
    np.testing.assert_array_equal(simp_verts_list[1], [[3.0, 3.0], [5.0, 5.0]])
    np.testing.assert_array_equal(simp_endpts_list[0], [[0, 1]])
    np.testing.assert_array_equal(simp_endpts_list[1], [[0, 1]])

    # Tuple of transposed/differing shapes
    verts0_t = vs0.T  # shape (2, 3)
    endpts0_t = endpts0.T  # shape (2, 1)

    simp_orders_tuple, simp_vs_tuple, simp_endpts_tuple, keeps_tuple = (
        graphs_m.simplify_flowgraph(
            *((orders0, orders1), (verts0_t, verts1), (endpts0_t, endpts1)),
            tol=1.0,
            check_topology=False,
            backend="fortran",
        )
    )
    assert isinstance(simp_orders_tuple, tuple)
    assert isinstance(simp_vs_tuple, tuple)
    np.testing.assert_array_equal(keeps_tuple[0], [T, F, T])
    np.testing.assert_array_equal(keeps_tuple[1], [T, F, T])
    # Verify orientation restoration
    assert simp_vs_tuple[0].shape == (2, 2)
    assert simp_endpts_tuple[0].shape == (2, 1)
    assert simp_vs_tuple[1].shape == (2, 2)
    assert simp_endpts_tuple[1].shape == (1, 2)


def test_simplify_multiple_flowgraphs_accepts_one_empty_graph():
    empty_orders = np.empty((0,), dtype=np.uint8)
    empty_verts = np.empty((0, 2), dtype=np.float32)
    empty_endpts = np.empty((0, 2), dtype=np.int32)
    orders = np.array([1], dtype=np.uint8)
    verts = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    endpts = np.array([[0, 1]], dtype=np.int32)

    simp_orders, simp_verts, simp_endpts, keeps = graphs_m.simplify_flowgraph(
        [empty_orders, orders],
        [empty_verts, verts],
        [empty_endpts, endpts],
        tol=0.0,
        check_topology=True,
        backend="fortran",
    )

    np.testing.assert_array_equal(simp_orders[0], empty_orders)
    np.testing.assert_array_equal(simp_verts[0], empty_verts)
    np.testing.assert_array_equal(simp_endpts[0], empty_endpts)
    np.testing.assert_array_equal(keeps[0], np.empty((0,), dtype=bool))
    np.testing.assert_array_equal(simp_orders[1], orders)
    np.testing.assert_array_equal(simp_verts[1], verts)
    np.testing.assert_array_equal(simp_endpts[1], endpts)
    np.testing.assert_array_equal(keeps[1], np.ones(2, dtype=bool))


@pytest.mark.parametrize("collection_type", (list, tuple))
def test_simplify_multiple_flowgraphs_round_trips_all_empty_graphs(collection_type):
    orders = collection_type(
        (np.empty((0,), dtype=np.uint8), np.empty((0,), dtype=np.int16))
    )
    verts = collection_type(
        (np.empty((0, 2), dtype=np.float32), np.empty((2, 0), dtype=np.float64))
    )
    endpts = collection_type(
        (np.empty((0, 2), dtype=np.int32), np.empty((2, 0), dtype=np.int64))
    )

    simp_orders, simp_verts, simp_endpts, keeps = graphs_m.simplify_flowgraph(
        orders,
        verts,
        endpts,
        tol=1.0,
        check_topology=True,
        backend="fortran",
    )

    assert isinstance(simp_orders, collection_type)
    assert isinstance(simp_verts, collection_type)
    assert isinstance(simp_endpts, collection_type)
    assert isinstance(keeps, collection_type)
    for original_group, simplified_group in zip(
        (orders, verts, endpts),
        (simp_orders, simp_verts, simp_endpts),
    ):
        for original, simplified in zip(original_group, simplified_group):
            np.testing.assert_array_equal(simplified, original)
            assert simplified.shape == original.shape
            assert simplified.dtype == original.dtype
    for keep in keeps:
        np.testing.assert_array_equal(keep, np.empty((0,), dtype=bool))


@pytest.mark.parametrize("collection_type", (list, tuple))
def test_simplify_multiple_flowgraphs_accepts_empty_collections(collection_type):
    empty = collection_type()

    results = graphs_m.simplify_flowgraph(empty, empty, empty)

    for result in results:
        assert isinstance(result, collection_type)
        assert len(result) == 0


def test_simplify_flowgraph_keeps_every_arc_endpoint():
    orders = np.array([1, 2, 3], dtype=np.uint8)
    verts = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.2],
            [2.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.2],
            [4.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
        ]
    )
    endpts = np.array([[0, 2], [3, 5], [6, 7]], dtype=np.int32)

    _, simp_verts, simp_endpts, keeps = graphs_m.simplify_flowgraph(
        orders,
        verts,
        endpts,
        tol=1.0,
        check_topology=True,
        backend="fortran",
    )

    assert np.all(keeps[endpts.ravel()])
    np.testing.assert_array_equal(
        simp_verts[simp_endpts.ravel()],
        verts[endpts.ravel()],
    )


def test_simplify_multiple_flowgraphs_inserts_overlap_endpoints():
    vertices = [
        np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        np.array([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]]),
    ]
    endpts = [np.array([[0, 2]]), np.array([[0, 2]])]
    orders = [np.array([3]), np.array([5])]

    simp_orders, simp_verts, simp_endpts, keeps = graphs_m.simplify_flowgraph(
        *(orders, vertices, endpts),
        tol=0.0,
        check_topology=True,
        backend="fortran",
    )
    np.testing.assert_array_equal(simp_orders[0], [3, 3])
    np.testing.assert_array_equal(simp_orders[1], [5, 5])

    expected_start_verts = [
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        np.array([[1.0, -1.0], [1.0, 0.0]]),
    ]
    for graph_verts, graph_endpts, graph_keeps, expected_starts in zip(
        simp_verts, simp_endpts, keeps, expected_start_verts
    ):
        assert graph_endpts.shape == (2, 2)
        np.testing.assert_array_equal(graph_endpts, np.array([[0, 1], [2, 3]]))
        np.testing.assert_array_equal(graph_verts[graph_endpts[:, 0]], expected_starts)
        # Batch overlap splitting rebuilds a compact vertex array, duplicating
        # only the shared endpoint required by the two resulting arcs.
        assert graph_keeps.size == 4

    # Both occurrences of the crossing are endpoints after each graph is split
    for graph_verts, graph_endpts in zip(simp_verts, simp_endpts):
        endpoint_verts = graph_verts[graph_endpts.ravel()]
        assert np.sum(np.all(endpoint_verts == [1.0, 0.0], axis=1)) == 2


def test_simplify_multiple_flowgraphs_ignores_identical_arcs():
    verts = [
        np.array([[0.0, 0.0], [1.0, 0.2], [2.0, 0.4], [3.0, 0.2], [4.0, 0.0]]),
        np.array([[4.0, 0.0], [3.0, 0.2], [2.0, 0.4], [1.0, 0.2], [0.0, 0.0]]),
    ]
    endpts = [np.array([[0, 4]]), np.array([[0, 4]])]
    orders = [np.array([2]), np.array([4])]

    _, simp_verts, _, _ = graphs_m.simplify_flowgraph(
        *(orders, verts, endpts),
        tol=0.25,
        check_topology=True,
        backend="fortran",
    )

    # The identical central arcs may simplify despite having opposite directions
    for graph_verts in simp_verts:
        assert not np.any(np.all(graph_verts == [2.0, 0.4], axis=1))


def test_flowgraph_3x3():
    dir_scheme = D8Directions(transform_codes=lambda x: x)

    # Config 1
    dirs = np.array([[3, 3, 3], [3, 3, 3], [1, 1, 0]], dtype=np.uint8, order="F")

    expected_fg_i = np.array(
        [
            *(0, 1, np.nan, 0, 1, np.nan, 0, 1, np.nan),
            *(1, 2, np.nan, 1, 2, np.nan, 1, 2, np.nan),
            *(2, 2, np.nan, 2, 2, np.nan, 2, 2, np.nan),
        ]
    )
    expected_fg_j = np.array(
        [
            *(0, 0, np.nan, 1, 1, np.nan, 2, 2, np.nan),
            *(0, 0, np.nan, 1, 1, np.nan, 2, 2, np.nan),
            *(0, 1, np.nan, 1, 2, np.nan, 2, 2, np.nan),
        ]
    )
    fg_i, fg_j = graphs_m.create_flowgraph(dirs, dir_scheme=dir_scheme)
    np.testing.assert_array_equal(fg_i, expected_fg_i)
    np.testing.assert_array_equal(fg_j, expected_fg_j)

    # Config 2
    dirs = np.array([[5, 1, 1], [5, 1, 1], [5, 1, 1]])
    expected_fg_i = np.array(
        [
            *(0, 0, np.nan),
            *(1, 1, np.nan),
            *(2, 2, np.nan),
        ]
    )
    expected_fg_j = np.array(
        [
            *(1, 2, np.nan),
            *(1, 2, np.nan),
            *(1, 2, np.nan),
        ]
    )
    with pytest.warns(UserWarning):
        fg_i, fg_j = graphs_m.create_flowgraph(dirs, dir_scheme=dir_scheme)
    np.testing.assert_array_equal(fg_i, expected_fg_i)
    np.testing.assert_array_equal(fg_j, expected_fg_j)
