# Last modified
#   2026-08-03, En-Chi Lee (williameclee@gmail.com)
#     - Added test cases for function `find_acyclic_flowdirs` and
#       graph construction validity.

import numpy as np
import pytest

from formosa import D8Directions
import formosa.geomorphology.flowdir.flowdir as flowdir_m
from formosa.geomorphology.flowdir.raster import raster as raster_m
from formosa.geomorphology.flowdir.graphs import graphs as graphs_m

T = True
F = False

BACKENDS = ("python", "fortran")


@pytest.fixture
def unequal_tributary_network():
    """A second-order branch joins a longer first-order branch."""
    dirs = np.zeros((4, 5), dtype=np.uint8)
    valids = np.zeros_like(dirs, dtype=bool)

    paths = {
        (0, 0): 2,  # southeast to (1, 1)
        (0, 2): 4,  # southwest to (1, 1)
        (1, 1): 3,  # south
        (2, 1): 3,  # south to the confluence
        (0, 4): 3,  # start of the longer first-order branch
        (1, 4): 4,
        (2, 3): 4,
        (3, 2): 5,
        (3, 1): 0,  # sink
    }
    for ij, direction in paths.items():
        dirs[ij] = direction
        valids[ij] = True

    expected = np.zeros_like(dirs, dtype=np.uint8)
    expected[valids] = 1
    expected[1, 1] = 2
    expected[2, 1] = 2
    expected[3, 1] = 2
    return dirs, valids, expected


@pytest.mark.parametrize("backend", ["python", "fortran"])
def test_unequal_tributary_does_not_increase_order(unequal_tributary_network, backend):
    dirs, valids, expected = unequal_tributary_network
    dir_scheme = D8Directions(transform_codes=lambda x: x)

    orders = raster_m.compute_flow_strahler_order(
        dirs,
        dir_scheme=dir_scheme,
        valids=valids,
        backend=backend,
    )

    np.testing.assert_array_equal(orders, expected)


def test_strahler_backends_match_with_mask_and_supplied_indegrees(
    unequal_tributary_network,
):
    dirs, valids, _ = unequal_tributary_network
    dir_scheme = D8Directions(transform_codes=lambda x: x)
    indegs = flowdir_m.count_indegree(
        dirs, dir_scheme=dir_scheme, valids=valids, backend="python"
    )
    original_indegs = indegs.copy()

    python_orders = raster_m.compute_flow_strahler_order(
        dirs,
        dir_scheme=dir_scheme,
        valids=valids,
        indegs=indegs,
        backend="python",
    )
    fortran_orders = raster_m.compute_flow_strahler_order(
        dirs,
        dir_scheme=dir_scheme,
        valids=valids,
        indegs=indegs,
        backend="fortran",
    )

    np.testing.assert_array_equal(python_orders, fortran_orders)
    np.testing.assert_array_equal(indegs, original_indegs)
    assert np.all(python_orders[~valids] == 0)


@pytest.mark.parametrize("backend", ["python", "fortran"])
def test_masked_tributary_does_not_affect_order(backend):
    dir_scheme = D8Directions(transform_codes=lambda x: x)
    dirs = np.array(
        [
            [2, 0, 4],
            [0, 3, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )
    valids = np.array(
        [
            [T, F, F],
            [F, T, F],
            [F, T, F],
        ]
    )
    expected = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=np.uint8,
    )

    orders = raster_m.compute_flow_strahler_order(
        dirs,
        dir_scheme=dir_scheme,
        valids=valids,
        backend=backend,
    )

    np.testing.assert_array_equal(orders, expected)


def test_construct_flowgraph_is_backend_independent_of_masked_directions():
    dir_scheme = D8Directions(transform_codes=lambda x: x)
    dirs = np.array([[3, 3, 3], [3, 3, 3], [1, 1, 0]], dtype=np.uint8)
    valids = np.array([[True, False, True], [True, True, True], [True, True, True]])
    changed_dirs = dirs.copy()
    changed_dirs[~valids] = 8

    results = []
    for backend in ("python", "fortran"):
        for candidate_dirs in (dirs, changed_dirs):
            arc_orders, vertex_ijs, arc_endpts = graphs_m.construct_flowgraph(
                candidate_dirs,
                dir_scheme=dir_scheme,
                valids=valids,
                min_order=1,
                backend=backend,
            )
            arcs = [
                (
                    int(order),
                    tuple(map(tuple, vertex_ijs[start : end + 1].tolist())),
                )
                for order, (start, end) in zip(arc_orders, arc_endpts)
            ]
            results.append(arcs)

    for actual in results[1:]:
        assert actual == results[0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_constructed_flowgraph_segments_follow_d8_adjacency(backend):
    dir_scheme = D8Directions(transform_codes=lambda x: x)
    dirs = np.array([[3, 3, 3], [3, 3, 3], [1, 1, 0]], dtype=np.uint8)
    valids = np.array([[T, F, T], [T, T, T], [T, T, T]])

    _, vertex_ijs, arc_endpts = graphs_m.construct_flowgraph(
        dirs,
        dir_scheme=dir_scheme,
        valids=valids,
        min_order=1,
        backend=backend,
    )

    for start, end in arc_endpts:
        arc = vertex_ijs[start : end + 1]
        offsets = np.diff(arc, axis=0)
        assert np.all(np.max(np.abs(offsets), axis=1) == 1)
        assert not np.any(np.all(offsets == 0, axis=1))
        for (i, j), offset in zip(arc[:-1], offsets):
            np.testing.assert_array_equal(
                offset,
                dir_scheme.code2d8offset(dirs[i, j]),
            )


@pytest.mark.parametrize("backend", BACKENDS)
def test_construct_flowgraph_rejects_two_cell_cycle(backend):
    dir_scheme = D8Directions(transform_codes=lambda x: x)
    dirs = np.array([[1, 5]], dtype=np.uint8)
    orders = np.ones(dirs.shape, dtype=np.uint8)

    with pytest.raises(graphs_m.DirectedFlowCycleError) as exc_info:
        graphs_m.construct_flowgraph(
            dirs,
            dir_scheme=dir_scheme,
            orders=orders,
            min_order=1,
            backend=backend,
        )

    np.testing.assert_array_equal(exc_info.value.cycle_ijs, [[0, 0], [0, 1]])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("preserve_junctions", (True, False))
def test_construct_flowgraph_reports_cycle_without_acyclic_feeder(
    backend,
    preserve_junctions,
):
    dir_scheme = D8Directions(transform_codes=lambda x: x)
    # Cell 0 feeds the cycle between cells 1 and 2.
    dirs = np.array([[1, 1, 5]], dtype=np.uint8)
    orders = np.ones(dirs.shape, dtype=np.uint8)

    with pytest.raises(graphs_m.DirectedFlowCycleError) as exc_info:
        graphs_m.construct_flowgraph(
            dirs,
            dir_scheme=dir_scheme,
            orders=orders,
            min_order=1,
            preserve_junctions=preserve_junctions,
            backend=backend,
        )

    np.testing.assert_array_equal(exc_info.value.cycle_ijs, [[0, 1], [0, 2]])


@pytest.mark.parametrize("backend", BACKENDS)
def test_construct_flowgraph_reports_disconnected_cycles(backend):
    dir_scheme = D8Directions(transform_codes=lambda x: x)
    dirs = np.array([[1, 5], [1, 5]], dtype=np.uint8)
    orders = np.ones(dirs.shape, dtype=np.uint8)

    with pytest.raises(graphs_m.DirectedFlowCycleError) as exc_info:
        graphs_m.construct_flowgraph(
            dirs,
            dir_scheme=dir_scheme,
            orders=orders,
            min_order=1,
            backend=backend,
        )

    np.testing.assert_array_equal(
        exc_info.value.cycle_ijs,
        [[0, 0], [0, 1], [1, 0], [1, 1]],
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_construct_flowgraph_allows_isolated_noflow_cell(backend):
    dir_scheme = D8Directions(transform_codes=lambda x: x)
    dirs = np.array([[0]], dtype=np.uint8)
    orders = np.ones(dirs.shape, dtype=np.uint8)

    graph_orders, graph_verts, graph_endpts = graphs_m.construct_flowgraph(
        dirs,
        dir_scheme=dir_scheme,
        orders=orders,
        min_order=1,
        backend=backend,
    )

    assert graph_orders.shape == (0,)
    assert graph_verts.shape == (0, 2)
    assert graph_endpts.shape == (0, 2)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    ("valids", "orders", "min_order"),
    [
        (np.array([[True, False]]), np.array([[1, 1]]), 1),
        (np.array([[True, True]]), np.array([[2, 1]]), 2),
    ],
)
def test_construct_flowgraph_allows_selection_boundary(
    backend,
    valids,
    orders,
    min_order,
):
    dir_scheme = D8Directions(transform_codes=lambda x: x)
    dirs = np.array([[1, 0]], dtype=np.uint8)

    graph_orders, graph_verts, graph_endpts = graphs_m.construct_flowgraph(
        dirs,
        dir_scheme=dir_scheme,
        valids=valids,
        orders=orders,
        min_order=min_order,
        backend=backend,
    )

    assert graph_orders.shape == (0,)
    assert graph_verts.shape == (0, 2)
    assert graph_endpts.shape == (0, 2)


@pytest.mark.parametrize("backend", BACKENDS)
def test_construct_flowgraph_covers_every_selected_edge_endpoint(backend):
    dir_scheme = D8Directions(transform_codes=lambda x: x)
    dirs = np.array([[3, 3, 3], [3, 3, 3], [1, 1, 0]], dtype=np.uint8)
    valids = np.array([[T, F, T], [T, T, T], [T, T, T]])
    orders = np.ones(dirs.shape, dtype=np.uint8)

    _, graph_verts, graph_endpts = graphs_m.construct_flowgraph(
        dirs,
        dir_scheme=dir_scheme,
        valids=valids,
        orders=orders,
        min_order=1,
        backend=backend,
    )

    represented = {
        tuple(ij) for start, end in graph_endpts for ij in graph_verts[start : end + 1]
    }
    expected = {
        (0, 0),
        (1, 0),
        (2, 0),
        (0, 2),
        (1, 2),
        (2, 2),
        (1, 1),
        (2, 1),
    }
    assert represented == expected
