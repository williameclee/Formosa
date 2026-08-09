# Last modified
#   2026-07-29, En-Chi Lee (williameclee@gmail.com)
#     - Added test cases for function `simplify_flowgraph`.

from tests.core import *

import pytest
import warnings
import numpy as np

import formosa.geomorphology.flowdir.network as nwork_m
import formosa.geomorphology.flowdir.network.validation as val_m
from formosa.geomorphology.flowdir.network import simplification as simp_m

from types import SimpleNamespace


def test_topology_repair_simplifies_each_conflicting_arc_once(
    monkeypatch: pytest.MonkeyPatch,
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

    def fake_locator(vertex_xys, arc_endpts, backend="fortran"):
        return next(locator_results)

    def fake_simplify(vertex_xys, arc_endpts, tol):
        simplified_starts.append(tuple(vertex_xys[:, 0]))
        return np.ones(vertex_xys.shape[1], dtype=np.int8)

    monkeypatch.setattr(simp_m, "locate_invalid_graph_topology", fake_locator)
    monkeypatch.setattr(
        simp_m,
        "simp_f",
        SimpleNamespace(simplify_flowgraph=fake_simplify),
    )

    vertices = np.array([[0, 1, 2, 3, 4, 5], [0, 0, 0, 0, 0, 0]], dtype=np.float32)
    endpts = np.array([[0, 2, 4], [1, 3, 5]], dtype=np.int32)
    keeps = simp_m._resolve_topology_intersections(
        vertices, endpts, np.ones(6, dtype=bool), tol=1.0, max_iters=1
    )

    assert simplified_starts == [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)]
    np.testing.assert_array_equal(keeps, np.ones(6, dtype=bool))


def test_topology_repair_attempt_count_matches_max_iters(
    monkeypatch: pytest.MonkeyPatch,
):
    intersections = np.array([[0, 1, 0, 2, 1]], dtype=np.int32)
    simplify_calls = []

    monkeypatch.setattr(
        simp_m,
        "locate_invalid_graph_topology",
        lambda *args, **kwargs: intersections,
    )

    def fake_simplify(vertex_xys, arc_endpts, tol):
        simplify_calls.append(tol)
        return np.ones(vertex_xys.shape[1], dtype=np.int8)

    monkeypatch.setattr(
        simp_m,
        "simp_f",
        SimpleNamespace(simplify_flowgraph=fake_simplify),
    )

    vertices = np.array([[0, 1, 2, 3], [0, 0, 0, 0]], dtype=np.float32)
    endpts = np.array([[0, 2], [1, 3]], dtype=np.int32)
    simp_m._resolve_topology_intersections(
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
    simp_orders, simp_verts, simp_endpts, keeps = simp_m.simplify_flowgraph(
        np.array([3]), verts, endpts, tol=1.0, check_topology=False, backend="fortran"
    )
    np.testing.assert_array_equal(simp_orders, [3])
    np.testing.assert_array_equal(keeps, [T, F, T])
    np.testing.assert_array_equal(simp_verts, [[0.0, 0.0], [2.0, 2.0]])
    np.testing.assert_array_equal(simp_endpts, [[0, 1]])

    # Test topology correction
    verts_topo = np.array([[0.0, 0.8], [1.0, 2.0], [2.0, 0.2], [0.5, 0.5], [1.5, 0.5]])
    endpts_topo = np.array([[0, 2], [3, 4]])

    # Under tol = 1.5 and check_topology = False, simplification occurs and causes intersection
    _, _, _, keeps_no_check = simp_m.simplify_flowgraph(
        *(orders, verts_topo, endpts_topo),
        tol=1.5,
        check_topology=False,
        backend="fortran",
    )
    # Vertex 1 should be removed
    np.testing.assert_array_equal(keeps_no_check, [T, F, T, T, T])

    # Under tol = 1.5 and check_topology = True, it detects intersection, reduces tolerance,
    # and keeps Vertex 1 to avoid intersection
    _, checked_verts, checked_endpts, keeps_with_check = simp_m.simplify_flowgraph(
        *(orders, verts_topo, endpts_topo),
        tol=1.5,
        check_topology=True,
        backend="fortran",
    )
    # Vertex 1 should be kept
    np.testing.assert_array_equal(keeps_with_check, [T, T, T, T, T])
    assert (
        val_m.locate_invalid_graph_topology(
            checked_verts, checked_endpts, backend="fortran"
        )
        is None
    )

    with warnings.catch_warnings():
        warnings.simplefilter("default")
        filters_before = list(warnings.filters)
        simp_m.simplify_flowgraph(
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

    simp_orders, simp_verts, simp_endpts, keeps = simp_m.simplify_flowgraph(
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
        simp_m.simplify_flowgraph(
            np.array([1]), "not-an-array", endpts, check_topology=False  # type: ignore
        )

    with pytest.raises(ValueError, match="Order array has length 0"):
        simp_m.simplify_flowgraph(
            np.array([], dtype=np.uint8), verts, endpts, check_topology=False
        )

    with pytest.raises(ValueError, match="must have the same length"):
        simp_m.simplify_flowgraph(orders, [verts], [endpts], check_topology=False)


def test_simplify_rejects_invalid_final_graph_from_valid_input(monkeypatch):
    verts = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [0.5, 0.5], [1.5, 0.5]])
    endpts = np.array([[0, 2], [3, 4]])
    orders = np.array([1, 2])
    final_intxs = np.array([[0, 1, 0, 2, 1]], dtype=np.int32)

    monkeypatch.setattr(
        simp_m,
        "_resolve_topology_intersections",
        lambda verts, endpts, keeps, tol, graph_ids=None: np.array([T, F, T, T, T]),
    )
    locator_results = iter([final_intxs, None])
    monkeypatch.setattr(
        simp_m,
        "_locate_disallowed_graph_topology",
        lambda verts, endpts, graph_ids=None: next(locator_results),
    )

    with pytest.raises(nwork_m.UnresolvedSimplificationTopology) as exc_info:
        simp_m.simplify_flowgraph(
            orders, verts, endpts, tol=1.0, check_topology=True, backend="fortran"
        )


def test_simplify_rejects_invalid_final_graph_from_invalid_input(
    monkeypatch: pytest.MonkeyPatch,
):
    verts = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    endpts = np.array([[0, 1], [2, 3]])
    orders = np.array([1, 2])
    final_intxs = np.array([[0, 1, 0, 2, 1]], dtype=np.int32)
    input_intxs = np.array([[0, 1, 0, 2, 1]], dtype=np.int32)

    monkeypatch.setattr(
        simp_m,
        "_resolve_topology_intersections",
        lambda verts, endpts, keeps, tol, graph_ids=None: np.ones(
            verts.shape[1], dtype=bool
        ),
    )
    locator_results = iter([final_intxs, input_intxs])
    monkeypatch.setattr(
        simp_m,
        "_locate_disallowed_graph_topology",
        lambda verts, endpts, graph_ids=None: next(locator_results),
    )

    with pytest.raises(nwork_m.InvalidOriginalGraphTopology) as exc_info:
        simp_m.simplify_flowgraph(
            orders, verts, endpts, tol=1.0, check_topology=True, backend="fortran"
        )


def test_simplify_skips_final_validation_when_topology_check_is_disabled(
    monkeypatch,
):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("Topology validation should be disabled.")

    monkeypatch.setattr(simp_m, "_locate_disallowed_graph_topology", fail_if_called)

    simp_m.simplify_flowgraph(
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
        simp_m.simplify_flowgraph(
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
        simp_m.simplify_flowgraph(
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

    simp_orders, simp_verts, simp_endpts, keeps = simp_m.simplify_flowgraph(
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

    simp_orders, simp_verts, simp_endpts, keeps = simp_m.simplify_flowgraph(
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

    results = simp_m.simplify_flowgraph(empty, empty, empty)

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

    _, simp_verts, simp_endpts, keeps = simp_m.simplify_flowgraph(
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

    simp_orders, simp_verts, simp_endpts, keeps = simp_m.simplify_flowgraph(
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

    _, simp_verts, _, _ = simp_m.simplify_flowgraph(
        orders, verts, endpts, tol=0.25, check_topology=True, backend="fortran"
    )

    # The identical central arcs may simplify despite having opposite directions
    for graph_verts in simp_verts:
        assert not np.any(np.all(graph_verts == [2.0, 0.4], axis=1))
