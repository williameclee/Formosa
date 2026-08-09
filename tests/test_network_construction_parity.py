"""
Verifies flow-graph construction parity across Python and FORTRAN
backends.

Last modified: 2026-08-10, En-Chi Lee (williameclee@gmail.com)
"""

from tests.core import *

import pytest
import numpy as np

from formosa.utils import BACKENDS
from formosa import D8Directions
import formosa.geomorphology.drainage.network as nwork_m
import formosa.geomorphology.drainage.network.construction as constr_m


def test_construct_flowgraph_is_backend_independent_of_masked_directions():
    dir_scheme = D8Directions(transform_codes=lambda x: x)
    dirs = np.array([[3, 3, 3], [3, 3, 3], [1, 1, 0]], dtype=np.uint8)
    valids = np.array([[T, F, T], [T, T, T], [T, T, T]])
    changed_dirs = dirs.copy()
    changed_dirs[~valids] = 8

    results = []
    for backend in BACKENDS:
        for candidate_dirs in (dirs, changed_dirs):
            arc_orders, vertex_ijs, arc_endpts = constr_m.construct_flowgraph(
                candidate_dirs,
                dir_scheme=dir_scheme,
                valids=valids,
                min_order=1,
                backend=backend,
            )
            arcs = [
                (int(order), tuple(map(tuple, vertex_ijs[start : end + 1].tolist())))
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

    _, vertex_ijs, arc_endpts = constr_m.construct_flowgraph(
        dirs, dir_scheme=dir_scheme, valids=valids, min_order=1, backend=backend
    )

    for start, end in arc_endpts:
        arc = vertex_ijs[start : end + 1]
        offsets = np.diff(arc, axis=0)
        assert np.all(np.max(np.abs(offsets), axis=1) == 1)
        assert not np.any(np.all(offsets == 0, axis=1))
        for (i, j), offset in zip(arc[:-1], offsets):
            np.testing.assert_array_equal(offset, dir_scheme.code2d8offset(dirs[i, j]))


@pytest.mark.parametrize("backend", BACKENDS)
def test_construct_flowgraph_rejects_two_cell_cycle(backend):
    dir_scheme = D8Directions(transform_codes=lambda x: x)
    dirs = np.array([[1, 5]], dtype=np.uint8)
    orders = np.ones(dirs.shape, dtype=np.uint8)

    with pytest.raises(nwork_m.validation.DirectedFlowCycleError) as exc_info:
        constr_m.construct_flowgraph(
            dirs, dir_scheme=dir_scheme, orders=orders, min_order=1, backend=backend
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

    with pytest.raises(nwork_m.validation.DirectedFlowCycleError) as exc_info:
        constr_m.construct_flowgraph(
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

    with pytest.raises(nwork_m.validation.DirectedFlowCycleError) as exc_info:
        constr_m.construct_flowgraph(
            dirs, dir_scheme=dir_scheme, orders=orders, min_order=1, backend=backend
        )

    np.testing.assert_array_equal(
        exc_info.value.cycle_ijs, [[0, 0], [0, 1], [1, 0], [1, 1]]
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_construct_flowgraph_allows_isolated_noflow_cell(backend):
    dir_scheme = D8Directions(transform_codes=lambda x: x)
    dirs = np.array([[0]], dtype=np.uint8)
    orders = np.ones(dirs.shape, dtype=np.uint8)

    graph_orders, graph_verts, graph_endpts = constr_m.construct_flowgraph(
        dirs, dir_scheme=dir_scheme, orders=orders, min_order=1, backend=backend
    )

    assert graph_orders.shape == (0,)
    assert graph_verts.shape == (0, 2)
    assert graph_endpts.shape == (0, 2)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    ("valids", "orders", "min_order"),
    [
        (np.array([[T, F]]), np.array([[1, 1]]), 1),
        (np.array([[T, T]]), np.array([[2, 1]]), 2),
    ],
)
def test_construct_flowgraph_allows_selection_boundary(
    backend, valids, orders, min_order
):
    dir_scheme = D8Directions(transform_codes=lambda x: x)
    dirs = np.array([[1, 0]], dtype=np.uint8)

    graph_orders, graph_verts, graph_endpts = constr_m.construct_flowgraph(
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

    _, graph_verts, graph_endpts = constr_m.construct_flowgraph(
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
    expected = {(0, 0), (1, 0), (2, 0), (0, 2), (1, 2), (2, 2), (1, 1), (2, 1)}
    assert represented == expected
