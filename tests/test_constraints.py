"""Tests construction and validation of planar constraint graphs."""

import numpy as np

from formosa.geomorphology.drainage.network.models import FlowGraph
from formosa.geomorphology.meshing.constraints import ConstraintGraph, ConstraintInput
from formosa.geomorphology.meshing.core import ConstraintKind


def _edge_kind(graph: ConstraintGraph, a: tuple[int, int], b: tuple[int, int]) -> int:
    coord_to_id = {tuple(coord): i for i, coord in enumerate(graph.indices)}
    edge = np.sort([coord_to_id[a], coord_to_id[b]])
    edge_id = np.flatnonzero(np.all(graph.edges == edge, axis=1))
    assert edge_id.size == 1
    return int(graph.edge_kinds[edge_id[0]])


def test_rectangular_boundary_uses_only_its_corners_when_uninterrupted():
    graph = ConstraintGraph([], shape=(5, 7))

    assert graph.indices.shape == (4, 2)
    assert graph.edges.shape == (4, 2)
    assert np.all(graph.edge_kinds == int(ConstraintKind.BOUNDARY))


def test_boundary_edges_split_at_existing_vertices_and_merge_kinds():
    valley = FlowGraph(
        np.array([[4, 1], [4, 5]], dtype=np.int32),
        np.array([[0, 1]], dtype=np.int32),
    )
    ridge = FlowGraph(
        np.array([[3, 3], [4, 3]], dtype=np.int32),
        np.array([[0, 1]], dtype=np.int32),
    )

    graph = ConstraintGraph(
        (
            ConstraintInput(valley, ConstraintKind.VALLEY),
            ConstraintInput(ridge, ConstraintKind.RIDGE),
        ),
        shape=(5, 7),
    )

    combined = int(ConstraintKind.VALLEY | ConstraintKind.BOUNDARY)
    assert _edge_kind(graph, (4, 1), (4, 3)) == combined
    assert _edge_kind(graph, (4, 3), (4, 5)) == combined
    assert _edge_kind(graph, (3, 3), (4, 3)) == int(ConstraintKind.RIDGE)
    assert _edge_kind(graph, (4, 0), (4, 1)) == int(ConstraintKind.BOUNDARY)
    assert _edge_kind(graph, (4, 5), (4, 6)) == int(ConstraintKind.BOUNDARY)
