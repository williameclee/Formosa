"""
Tests unconstrained triangulation using the FORTRAN backend.

This module covers native coordinate-type and range validation by
the public meshing API.

Created: 2026-08-12, En-Chi Lee (williameclee@gmail.com)
"""

import pytest

import numpy as np

from formosa.geomorphology.meshing import triangulation as tri_m


def test_fortran_triangulation_rejects_float_coordinates():
    vtxs = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])

    with pytest.raises(TypeError, match="requires integer coordinates"):
        tri_m.triangulate_points(vtxs, backend="fortran")


def test_fortran_triangulation_rejects_coordinates_outside_int32():
    vtxs = np.array(
        [[0, 0], [0, 1], [np.iinfo(np.int32).max + 1, 0]],
        dtype=np.int64,
    )

    with pytest.raises(OverflowError, match="representable as int32"):
        tri_m.triangulate_points(vtxs, backend="fortran")


def test_fortran_triangulation_accepts_uint32_inside_int32_range():
    vtxs = np.array([[0, 0], [0, 4], [4, 0]], dtype=np.uint32)

    triangles = tri_m.triangulate_points(vtxs, backend="fortran")  # type: ignore

    assert triangles.shape == (1, 3)
    assert triangles.dtype == np.int32


def test_fortran_triangulation_rejects_uint32_outside_int32_range():
    vtxs = np.array(
        [[0, 0], [0, 1], [np.iinfo(np.int32).max + 1, 0]],
        dtype=np.uint32,
    )

    with pytest.raises(OverflowError, match="representable as int32"):
        tri_m.triangulate_points(vtxs, backend="fortran")  # type: ignore
