"""
Tests the public unconstrained Delaunay triangulation API.

This module covers point triangulation and facet-neighbour behaviour
shared by the Python and Fortran backends, plus native input
validatio and error translation. Constrained edge recovery is tested
separately in `test_triangulation_constrained.py`.

Created: 2026-08-12, En-Chi Lee (williameclee@gmail.com)
Last modified: 2026-08-17, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np
import pytest

from formosa.geomorphology.drainage.network import GraphTopologyError
from formosa.geomorphology.geometry import incircle, orient
from formosa.geomorphology.meshing import triangulation as tri_m
from formosa.utils import BACKENDS


def _mesh_edges(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.vstack(
        (triangles[:, (0, 1)], triangles[:, (1, 2)], triangles[:, (2, 0)])
    )
    edges.sort(axis=1)
    return np.unique(edges, axis=0, return_counts=True)


def _coordinate_edges(
    vtxs: np.ndarray, triangles: np.ndarray
) -> set[tuple[tuple[int, int], tuple[int, int]]]:
    edges, _ = _mesh_edges(triangles)
    return {tuple(sorted((tuple(vtxs[u]), tuple(vtxs[v])))) for u, v in edges}  # type: ignore


def _assert_valid_delaunay(vtxs: np.ndarray, triangles: np.ndarray) -> None:
    assert triangles.ndim == 2
    assert triangles.shape[1] == 3
    assert triangles.dtype == np.int32
    assert np.all(triangles >= 0)
    assert np.all(triangles < vtxs.shape[0])
    assert np.unique(np.sort(triangles, axis=1), axis=0).shape == triangles.shape

    used_vertices = np.unique(triangles)
    np.testing.assert_array_equal(used_vertices, np.arange(vtxs.shape[0]))

    for triangle in triangles:
        a, b, c = triangle
        assert orient(vtxs[a], vtxs[b], vtxs[c], backend="python") > 0
        other_ids = np.setdiff1d(np.arange(vtxs.shape[0]), triangle)
        for point_id in other_ids:
            assert (
                incircle(
                    vtxs[a],
                    vtxs[b],
                    vtxs[c],
                    vtxs[point_id],
                    oriented=True,
                    backend="python",
                )
                <= 0
            )

    _, incidence = _mesh_edges(triangles)
    assert np.all((incidence == 1) | (incidence == 2))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "points",
    [
        np.array([[0, 0], [0, 4], [4, 0]], dtype=np.int32),
        np.array([[0, 0], [0, 2], [2, 0], [2, 2]], dtype=np.int32),
        np.array([[0, 0], [0, 4], [4, 0], [4, 4], [2, 2]], dtype=np.int32),
        np.array([[1, 1], [2, 7], [4, 3], [6, 9], [8, 2], [9, 6]], dtype=np.int32),
        np.array([[10, 7], [11, 0], [11, 1], [11, 11]], dtype=np.int32),
    ],
)
def test_triangulate_points_produces_valid_delaunay_mesh(points, backend):
    triangles = tri_m.triangulate_points(points, backend=backend)
    _assert_valid_delaunay(points, triangles)


@pytest.mark.parametrize("backend", BACKENDS)
def test_triangulate_points_returns_canonical_triangle_order(backend):
    vtxs = np.array(
        [[0, 0], [1, 5], [3, 2], [5, 7], [8, 1], [9, 6], [4, 4]],
        dtype=np.int32,
    )

    triangles = tri_m.triangulate_points(vtxs, backend=backend)

    assert np.all(triangles[:, 0] == np.min(triangles, axis=1))
    order = np.lexsort((triangles[:, 2], triangles[:, 1], triangles[:, 0]))
    np.testing.assert_array_equal(order, np.arange(triangles.shape[0]))


def test_triangulate_points_backend_order_is_identical():
    vtxs = np.array(
        [[0, 0], [1, 5], [3, 2], [5, 7], [8, 1], [9, 6], [4, 4]],
        dtype=np.int32,
    )

    np.testing.assert_array_equal(
        tri_m.triangulate_points(vtxs, backend="python"),
        tri_m.triangulate_points(vtxs, backend="fortran"),
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("shape", [(2, 2), (3, 3), (5, 5), (3, 7)])
def test_triangulate_regular_raster_grid(shape, backend):
    nrows, ncols = shape
    vtxs = np.indices(shape).reshape(2, -1).T.astype(np.int32)

    triangles = tri_m.triangulate_points(vtxs, backend=backend)

    assert triangles.shape == (2 * (nrows - 1) * (ncols - 1), 3)
    _assert_valid_delaunay(vtxs, triangles)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "vtxs",
    [
        np.array([[0, 0], [4, 0], [0, 4], [4, 4], [2, 0]], dtype=np.int32),
        np.array(
            [[0, 0], [1, 0], [2, 0], [3, 0], [0, 3], [3, 3]],
            dtype=np.int32,
        ),
        np.array(
            [[0, 0], [4, 0], [0, 4], [4, 4], [1, 1], [2, 2], [3, 3]],
            dtype=np.int32,
        ),
    ],
)
def test_triangulate_accepts_collinear_subsets(vtxs, backend):
    triangles = tri_m.triangulate_points(vtxs, backend=backend)

    _assert_valid_delaunay(vtxs, triangles)


@pytest.mark.parametrize("backend", BACKENDS)
def test_triangulation_is_invariant_to_input_order(backend):
    vtxs = np.array(
        [[0, 0], [1, 5], [3, 2], [5, 7], [8, 1], [9, 6], [4, 4]],
        dtype=np.int32,
    )
    expected_edges = _coordinate_edges(
        vtxs, tri_m.triangulate_points(vtxs, backend=backend)
    )
    rng = np.random.default_rng(20260812)

    for _ in range(10):
        permuted_vtxs = vtxs[rng.permutation(vtxs.shape[0])]
        triangles = tri_m.triangulate_points(permuted_vtxs, backend=backend)
        assert _coordinate_edges(permuted_vtxs, triangles) == expected_edges


@pytest.mark.parametrize("backend", BACKENDS)
def test_cocircular_permutations_produce_valid_triangulations(backend):
    vtxs = np.array([[0, 0], [0, 4], [4, 0], [4, 4]], dtype=np.int32)

    for permutation in (
        np.array([0, 1, 2, 3]),
        np.array([3, 2, 1, 0]),
        np.array([1, 3, 0, 2]),
    ):
        permuted_vtxs = vtxs[permutation]
        triangles = tri_m.triangulate_points(permuted_vtxs, backend=backend)
        _assert_valid_delaunay(permuted_vtxs, triangles)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("nvtxs", [10, 30, 100])
def test_triangulate_deterministic_random_raster_points(nvtxs, backend):
    rng = np.random.default_rng(20260812 + nvtxs)
    candidates = rng.integers(0, 10_000, size=(nvtxs * 2, 2), dtype=np.int32)
    vtxs = np.unique(candidates, axis=0)[:nvtxs]
    assert vtxs.shape[0] == nvtxs

    triangles = tri_m.triangulate_points(vtxs, backend=backend)

    _assert_valid_delaunay(vtxs, triangles)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("offset", [10_000, 20_123])
def test_triangulation_is_translation_invariant(offset, backend):
    vtxs = np.array(
        [[0, 0], [1, 5], [3, 2], [5, 7], [8, 1], [9, 6], [4, 4]],
        dtype=np.int32,
    )
    translated_vtxs = vtxs + offset

    triangles = tri_m.triangulate_points(vtxs, backend=backend)
    translated_triangles = tri_m.triangulate_points(translated_vtxs, backend=backend)

    np.testing.assert_array_equal(
        np.sort(_mesh_edges(triangles)[0], axis=0),
        np.sort(_mesh_edges(translated_triangles)[0], axis=0),
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("scale", [2, 7, 1_000])
def test_triangulation_is_scale_invariant(scale, backend):
    vtxs = np.array(
        [[0, 0], [1, 5], [3, 2], [5, 7], [8, 1], [9, 6], [4, 4]],
        dtype=np.int32,
    )

    triangles = tri_m.triangulate_points(vtxs, backend=backend)
    scaled_triangles = tri_m.triangulate_points(vtxs * scale, backend=backend)

    np.testing.assert_array_equal(
        np.sort(_mesh_edges(triangles)[0], axis=0),
        np.sort(_mesh_edges(scaled_triangles)[0], axis=0),
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "vtxs",
    [
        np.array(
            [
                [np.iinfo(np.int32).max - 7, 0],
                [np.iinfo(np.int32).max, 0],
                [np.iinfo(np.int32).max - 7, 7],
            ],
            dtype=np.int32,
        ),
        np.array(
            [[0, 0], [600_000_000, 0], [300_000_000, 1]],
            dtype=np.int32,
        ),
    ],
)
def test_triangulate_int32_extreme_coordinates(vtxs, backend):
    triangles = tri_m.triangulate_points(vtxs, backend=backend)

    _assert_valid_delaunay(vtxs, triangles)


@pytest.mark.parametrize("backend", BACKENDS)
def test_triangulate_rejects_duplicate_vertices(backend):
    points = np.array([[0, 0], [1, 0], [0, 1], [1, 0]], dtype=np.int32)

    with pytest.raises(ValueError, match="1 duplicates"):
        tri_m.triangulate_points(points, backend=backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_triangulate_rejects_collinear_vertices(backend):
    vtxs = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int32)

    with pytest.raises(GraphTopologyError):
        tri_m.triangulate_points(vtxs, backend=backend)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "vtxs",
    [
        np.array([0, 1], dtype=np.int32),
        np.zeros((3, 3), dtype=np.int32),
    ],
)
def test_triangulate_rejects_invalid_shapes(vtxs, backend):
    with pytest.raises(ValueError, match="shape"):
        tri_m.triangulate_points(vtxs, backend=backend)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("nvtxs", [0, 1, 2])
def test_triangulate_rejects_too_few_vertices(nvtxs, backend):
    vtxs = np.arange(nvtxs * 2, dtype=np.int32).reshape(nvtxs, 2)

    with pytest.raises(ValueError, match="At least 3 vertices"):
        tri_m.triangulate_points(vtxs, backend=backend)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", [np.bool_, np.str_, object])
def test_triangulate_rejects_non_numeric_coordinates(dtype, backend):
    vtxs = np.array([[0, 0], [0, 2], [2, 0]], dtype=dtype)

    with pytest.raises(TypeError):
        tri_m.triangulate_points(vtxs, backend=backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_triangulate_accepts_noncontiguous_vertex_array(backend):
    storage = np.zeros((6, 4), dtype=np.int32)
    storage[:, ::2] = np.array(
        [[0, 0], [0, 4], [4, 0], [4, 4], [1, 2], [3, 1]], dtype=np.int32
    )
    vtxs = storage[:, ::2]
    assert not vtxs.flags.c_contiguous

    faces = tri_m.triangulate_points(vtxs, backend=backend)
    _assert_valid_delaunay(vtxs, faces)


def test_triangulate_points_rejects_unknown_backend():
    vtxs = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int32)

    with pytest.raises(ValueError, match="Unknown backend"):
        tri_m.triangulate_points(vtxs, backend="unknown")  # type: ignore


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    ("faces", "exp_nabrs"),
    [
        (
            np.array([[0, 1, 2]], dtype=np.int32),
            np.array([[-1, -1, -1]], dtype=np.int32),
        ),
        (
            np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32),
            np.array([[1, -1, -1], [-1, 0, -1]], dtype=np.int32),
        ),
        (
            np.array([[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4]], dtype=np.int32),
            np.array(
                [[3, 1, -1], [-1, -1, 0], [-1, 3, -1], [-1, 0, 2]], dtype=np.int32
            ),
        ),
    ],
)
def test_find_facet_neighbours(faces, exp_nabrs, backend):
    nabrs = tri_m.find_facet_neighbours(faces, backend=backend)

    np.testing.assert_array_equal(nabrs, exp_nabrs)
    assert nabrs.dtype == np.int32
    assert nabrs.flags.c_contiguous


@pytest.mark.parametrize("backend", BACKENDS)
def test_find_facet_neighbours_accepts_noncontiguous_input(backend):
    storage = np.array([[0, 99, 1, 99, 2, 99], [1, 99, 3, 99, 2, 99]], dtype=np.int32)
    faces = storage[:, ::2]
    assert not faces.flags.c_contiguous

    nabrs = tri_m.find_facet_neighbours(faces, backend=backend)
    np.testing.assert_array_equal(nabrs, [[1, -1, -1], [-1, 0, -1]])


@pytest.mark.parametrize("backend", BACKENDS)
def test_find_facet_neighbours_rejects_invalid_shape(backend):
    with pytest.raises(ValueError, match="shape"):
        tri_m.find_facet_neighbours(np.array([0, 1, 2]), backend=backend)


def test_find_facet_neighbours_rejects_unknown_backend():
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    with pytest.raises(ValueError, match="Unknown backend"):
        tri_m.find_facet_neighbours(faces, backend="unknown")  # type: ignore
