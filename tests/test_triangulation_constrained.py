"""
Tests the public constrained triangulation API.

This module covers edge flipping and constraint-edge recovery shared
by the Python and Fortran backends. Unconstrained point
triangulation and facet-neighbour behaviour are tested separately in
`test_triangulation.py`.

Created: 2026-08-17, En-Chi Lee (williameclee@gmail.com)
"""

import pytest

import numpy as np

from formosa.utils import BACKENDS
from formosa.geomorphology.drainage.network import GraphTopologyError
from formosa.geomorphology.geometry import orient
from formosa.geomorphology.meshing import triangulation as tri_m
from tests.test_triangulation import _mesh_edges


def test_recover_constraint_edges_rejects_unknown_backend():
    vtxs = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.int32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    edges = np.empty((0, 2), dtype=np.int32)

    with pytest.raises(ValueError, match="Unknown backend"):
        tri_m.recover_constraint_edges(vtxs, faces, edges, backend="unknown")  # type: ignore


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "edges, error, message",
    [
        (np.array([0, 1], dtype=np.int32), ValueError, r"shape \(E, 2\)"),
        (np.array([[0.0, 1.0]]), TypeError, "must be integers"),
        (np.array([[-1, 1]], dtype=np.int32), IndexError, "non-negative"),
        (np.array([[0, 3]], dtype=np.int32), IndexError, "out of bounds"),
        (np.array([[1, 1]], dtype=np.int32), ValueError, "1 self-edges"),
    ],
)
def test_recover_constraint_edges_rejects_invalid_edge_arrays(
    backend, edges, error, message
):
    vtxs = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.int32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    with pytest.raises(error, match=message):
        tri_m.recover_constraint_edges(vtxs, faces, edges, backend=backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_recover_constraint_edges_reports_failing_edge_position(backend):
    vtxs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int32)
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    edges = np.array([[1, 2], [0, 3]], dtype=np.int32)

    with pytest.raises(GraphTopologyError, match=r"constraint edge 1 \(0, 3\)"):
        tri_m.recover_constraint_edges(vtxs, faces, edges, backend=backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_recover_constraint_edges_returns_canonical_facet_topology(backend):
    vtxs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int32)
    faces = np.array([[3, 2, 1], [2, 0, 1]], dtype=np.int32)
    edges = np.empty((0, 2), dtype=np.int32)

    r_faces, nabrs = tri_m.recover_constraint_edges(vtxs, faces, edges, backend=backend)

    np.testing.assert_array_equal(r_faces, [[0, 1, 2], [1, 3, 2]])
    np.testing.assert_array_equal(nabrs, [[1, -1, -1], [-1, 0, -1]])
    np.testing.assert_array_equal(
        nabrs, tri_m.find_facet_neighbours(r_faces, backend=backend)
    )
    assert r_faces.dtype == np.int32
    assert r_faces.flags.c_contiguous
    assert nabrs.dtype == np.int32
    assert nabrs.flags.c_contiguous


@pytest.mark.parametrize("backend", BACKENDS)
def test_recover_constraint_edges_accepts_empty_constraint_set(backend):
    vtxs = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.int32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    r_faces, nabrs = tri_m.recover_constraint_edges(
        vtxs, faces, np.empty((0, 2), dtype=np.int32), backend=backend
    )

    np.testing.assert_array_equal(r_faces, faces)
    np.testing.assert_array_equal(nabrs, [[-1, -1, -1]])


@pytest.mark.parametrize("backend", BACKENDS)
def test_recover_constraint_edges_recovers_and_preserves_every_edge(backend):
    vtxs = np.indices((2, 4)).reshape(2, -1).T.astype(np.int32)
    faces = np.array(
        [
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
        ],
        dtype=np.int32,
    )
    edges = np.array([[1, 4], [2, 5], [3, 6]], dtype=np.int32)

    r_faces, nabrs = tri_m.recover_constraint_edges(vtxs, faces, edges, backend=backend)

    mesh_edges, _ = _mesh_edges(r_faces)
    mesh_edge_set = {tuple(map(int, edge)) for edge in mesh_edges}
    assert {tuple(sorted(map(int, edge))) for edge in edges} <= mesh_edge_set
    assert np.all(r_faces[:, 0] == np.min(r_faces, axis=1))
    order = np.lexsort((r_faces[:, 2], r_faces[:, 1], r_faces[:, 0]))
    np.testing.assert_array_equal(order, np.arange(r_faces.shape[0]))
    np.testing.assert_array_equal(
        nabrs, tri_m.find_facet_neighbours(r_faces, backend=backend)
    )
    np.testing.assert_array_equal(
        faces, [[0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3]]
    )


def test_recover_constraint_edge_rejects_unknown_backend():
    vtxs = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.int32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    with pytest.raises(ValueError, match="Unknown backend"):
        tri_m.recover_constraint_edge(vtxs, faces, (0, 1), backend="unknown")  # type: ignore


@pytest.mark.parametrize("backend", BACKENDS)
def test_exterior_guard_makes_boundary_constraint_recoverable(backend):
    vtxs = np.array([[0, 0], [0, 4], [2, 2], [-2, 2]], dtype=np.int32)
    faces = np.array([[0, 2, 3], [1, 3, 2]], dtype=np.int32)

    r_faces, _ = tri_m.recover_constraint_edge(vtxs, faces, (0, 1), backend=backend)
    r_faces = r_faces[np.all(r_faces < 3, axis=1)]

    assert np.any(np.any(r_faces == 0, axis=1) & np.any(r_faces == 1, axis=1))
    np.testing.assert_array_equal(r_faces, [[0, 2, 1]])


@pytest.mark.parametrize("backend", BACKENDS)
def test_recover_constraint_edge_rejects_edge_outside_dented_mesh_boundary(backend):
    vtxs = np.array([[0, 0], [0, 4], [1, 2], [2, 2]], dtype=np.int32)
    faces = np.array([[0, 3, 2], [2, 3, 1]], dtype=np.int32)

    with pytest.raises(GraphTopologyError):
        tri_m.recover_constraint_edge(vtxs, faces, (0, 1), backend=backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_recover_constraint_edge_does_not_flip_locked_crossing_edge(backend):
    vtxs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int32)
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)

    with pytest.raises(GraphTopologyError):
        tri_m.recover_constraint_edge(
            vtxs, faces, (0, 3), locked_edges={(2, 1)}, backend=backend
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_recover_constraint_edge_is_noop_when_edge_exists(backend):
    vtxs = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.int32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    r_faces, nabrs = tri_m.recover_constraint_edge(vtxs, faces, (0, 1), backend=backend)

    np.testing.assert_array_equal(r_faces, faces)
    np.testing.assert_array_equal(nabrs, [[-1, -1, -1]])


@pytest.mark.parametrize("backend", BACKENDS)
def test_recover_constraint_edge_allows_neutral_progress_flip(backend):
    vtxs = np.array(
        [
            [10, 73],
            [12, 44],
            [14, 65],
            [14, 87],
            [26, 16],
            [26, 54],
            [28, 10],
            [30, 43],
        ],
        dtype=np.int32,
    )
    faces = np.array(
        [[0, 1, 2], [0, 2, 3], [2, 1, 5], [3, 2, 5], [1, 4, 7], [5, 1, 7], [4, 6, 7]],
        dtype=np.int32,
    )

    r_faces, _ = tri_m.recover_constraint_edge(vtxs, faces, (0, 6), backend=backend)

    assert np.any(np.any(r_faces == 0, axis=1) & np.any(r_faces == 6, axis=1))


@pytest.mark.parametrize("backend", BACKENDS)
def test_recover_constraint_edge_handles_multiple_crossings(backend):
    vtxs = np.indices((2, 4)).reshape(2, -1).T.astype(np.int32)
    faces = np.array(
        [
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
        ],
        dtype=np.int32,
    )

    r_faces, nabrs = tri_m.recover_constraint_edge(vtxs, faces, (0, 7), backend=backend)

    assert np.any(np.any(r_faces == 0, axis=1) & np.any(r_faces == 7, axis=1))
    assert all(
        orient(*(vtxs[vtx] for vtx in face), backend="python") > 0 for face in r_faces
    )
    np.testing.assert_array_equal(
        nabrs, tri_m.find_facet_neighbours(r_faces, backend=backend)
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_recover_constraint_edge_flips_crossing_diagonal(backend):
    vtxs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int32)
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    input_nabrs = tri_m.find_facet_neighbours(faces, backend=backend)
    exp_input_nabrs = input_nabrs.copy()

    r_faces, nabrs = tri_m.recover_constraint_edge(
        vtxs, faces, (0, 3), nabrs=input_nabrs, backend=backend
    )

    assert np.any(np.any(r_faces == 0, axis=1) & np.any(r_faces == 3, axis=1))
    assert r_faces.dtype == np.int32
    assert r_faces.flags.c_contiguous
    assert nabrs.dtype == np.int32
    assert nabrs.flags.c_contiguous
    np.testing.assert_array_equal(nabrs, [[-1, 1, -1], [-1, -1, 0]])
    np.testing.assert_array_equal(faces, [[0, 1, 2], [1, 3, 2]])
    np.testing.assert_array_equal(input_nabrs, exp_input_nabrs)


def test_find_crossing_edges_fortran_does_not_modify_inputs():
    vtxs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int32)
    faces = np.array([[0, 2, 1], [2, 3, 1]], dtype=np.int32)
    nabrs = tri_m.find_facet_neighbours(faces, backend="fortran")
    input_vtxs = vtxs.copy()
    input_faces = faces.copy()
    input_nabrs = nabrs.copy()

    xngs = tri_m._find_crossing_edges(vtxs, faces, nabrs, (0, 3), backend="fortran")

    assert xngs == [(0, 0, (1, 2))]
    np.testing.assert_array_equal(vtxs, input_vtxs)
    np.testing.assert_array_equal(faces, input_faces)
    np.testing.assert_array_equal(nabrs, input_nabrs)


@pytest.mark.parametrize("backend", BACKENDS)
def test_find_crossing_edges_preserves_mesh_order(backend):
    vtxs = np.indices((2, 4)).reshape(2, -1).T.astype(np.int32)
    faces = np.array(
        [
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
        ],
        dtype=np.int32,
    )
    nabrs = tri_m.find_facet_neighbours(faces, backend=backend)

    xngs = tri_m._find_crossing_edges(vtxs, faces, nabrs, (0, 7), backend)

    assert xngs == [
        (1, 0, (1, 5)),
        (2, 1, (1, 6)),
        (3, 0, (2, 6)),
    ]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("edge", [(0, 1), (0, 2), (0, 4)])
def test_find_crossing_edges_excludes_nonproper_intersections(backend, edge):
    vtxs = np.array([[0, 0], [0, 2], [2, 0], [2, 2], [1, 1]], dtype=np.int32)
    faces = np.array([[0, 2, 4], [2, 3, 4], [3, 1, 4], [1, 0, 4]], dtype=np.int32)
    nabrs = tri_m.find_facet_neighbours(faces, backend=backend)

    xngs = tri_m._find_crossing_edges(vtxs, faces, nabrs, edge, backend)
    assert xngs == []


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("scale", [1, 100_000])
def test_find_crossing_edges_returns_unique_canonical_edges(backend, scale):
    vtxs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int32) * scale
    faces = np.array([[0, 2, 1], [2, 3, 1]], dtype=np.int32)
    nabrs = tri_m.find_facet_neighbours(faces, backend=backend)

    xngs = tri_m._find_crossing_edges(vtxs, faces, nabrs, (0, 3), backend)
    assert xngs == [(0, 0, (1, 2))]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    ("vtxs", "faces"),
    [
        (
            np.array([[0, 1], [0, 0], [2, 0], [-1, -1]], dtype=np.int32),
            np.array([[0, 1, 2], [3, 2, 1]], dtype=np.int32),
        ),
        (
            np.array([[0, 1], [0, 0], [2, 0], [0, -1]], dtype=np.int32),
            np.array([[0, 1, 2], [3, 2, 1]], dtype=np.int32),
        ),
    ],
)
def test_flip_quadrilateral_edge_rejects_unflippable_quadrilateral(
    vtxs, faces, backend
):
    with pytest.raises(GraphTopologyError):
        tri_m.flip_quadrilateral_edge(vtxs, faces, 0, 0, backend=backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_flip_quadrilateral_edge_rejects_boundary_edge(backend):
    vtxs = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.int32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    with pytest.raises(GraphTopologyError):
        tri_m.flip_quadrilateral_edge(vtxs, faces, 0, 0, backend=backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_flip_quadrilateral_edge_is_topologically_reversible(backend):
    vtxs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int32)
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    f_faces, f_nabrs = tri_m.flip_quadrilateral_edge(vtxs, faces, 0, 0, backend=backend)

    r_faces, _ = tri_m.flip_quadrilateral_edge(
        vtxs, f_faces, 0, 1, nabrs=f_nabrs, backend=backend
    )

    np.testing.assert_array_equal(
        np.sort(np.sort(r_faces, axis=1), axis=0),
        np.sort(np.sort(faces, axis=1), axis=0),
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_flip_quadrilateral_edge_updates_outside_neighbours(backend):
    vtxs = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 0], [-1, 0]], dtype=np.int32)
    faces = np.array([[0, 1, 2], [1, 3, 2], [1, 4, 3], [5, 0, 2]], dtype=np.int32)
    nabrs = tri_m.find_facet_neighbours(faces, backend=backend)

    f_faces, f_nabrs = tri_m.flip_quadrilateral_edge(
        vtxs, faces, 0, 0, nabrs=nabrs, backend=backend
    )

    r_nabrs = tri_m.find_facet_neighbours(f_faces, backend=backend)
    np.testing.assert_array_equal(f_nabrs, r_nabrs)
    np.testing.assert_array_equal(
        f_nabrs,
        [[2, 1, -1], [-1, 3, 0], [-1, 0, -1], [1, -1, -1]],
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_flip_quadrilateral_edge_replaces_convex_quadrilateral_diagonal(backend):
    vtxs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int32)
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    nabrs = tri_m.find_facet_neighbours(faces, backend=backend)
    input_nabrs = nabrs.copy()

    f_faces, f_nabrs = tri_m.flip_quadrilateral_edge(
        vtxs, faces, 0, 0, nabrs=nabrs, backend=backend
    )

    np.testing.assert_array_equal(f_faces, [[0, 1, 3], [0, 3, 2]])
    np.testing.assert_array_equal(f_nabrs, [[-1, 1, -1], [-1, -1, 0]])
    np.testing.assert_array_equal(faces, [[0, 1, 2], [1, 3, 2]])
    np.testing.assert_array_equal(nabrs, input_nabrs)
