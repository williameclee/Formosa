"""
Tests the public unconstrained Delaunay triangulation API.

This module covers behaviour shared by both backends and native
input validation and error translation.

Created: 2026-08-12, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np
import pytest

from formosa.geomorphology.drainage.network import GraphTopologyError
from formosa.geomorphology.geometry import incircle, orient_v2
from formosa.geomorphology.meshing import triangulation as tri_m
from formosa.geomorphology.meshing._backends import triangulation_py as tri_py
from formosa.utils import BACKENDS


def _mesh_edges(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.vstack(
        (triangles[:, (0, 1)], triangles[:, (1, 2)], triangles[:, (2, 0)])
    )
    edges.sort(axis=1)
    return np.unique(edges, axis=0, return_counts=True)


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
        assert orient_v2(vtxs[a], vtxs[b], vtxs[c], backend="python") > 0
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


def test_python_supertriangle_matches_native_construction():
    vtxs = np.array([[-3, 2], [5, -4]], dtype=np.int32)

    all_vtxs, supertriangle = tri_py.add_supertriangle(vtxs)

    np.testing.assert_array_equal(all_vtxs[:2], vtxs)
    np.testing.assert_array_equal(
        all_vtxs[2:],
        [[-31, -17], [33, -17], [1, 31]],
    )
    assert supertriangle == (2, 3, 4)
    assert (
        orient_v2(
            *(all_vtxs[vertex_id] for vertex_id in supertriangle),
            backend="python",
        )
        > 0
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "points",
    [
        np.array([[0, 0], [0, 4], [4, 0]], dtype=np.int32),
        np.array([[0, 0], [0, 2], [2, 0], [2, 2]], dtype=np.int32),
        np.array([[0, 0], [0, 4], [4, 0], [4, 4], [2, 2]], dtype=np.int32),
        np.array(
            [[1, 1], [2, 7], [4, 3], [6, 9], [8, 2], [9, 6]],
            dtype=np.int32,
        ),
    ],
)
def test_triangulate_points_produces_valid_delaunay_mesh(points, backend):
    triangles = tri_m.triangulate_points(points, backend=backend)
    _assert_valid_delaunay(points, triangles)


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


def test_triangulate_points_rejects_unknown_backend():
    vtxs = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int32)

    with pytest.raises(ValueError, match="Unknown backend"):
        tri_m.triangulate_points(vtxs, backend="unknown")  # type: ignore
