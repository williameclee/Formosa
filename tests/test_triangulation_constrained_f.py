"""
Tests the conversion of Python's 0-based indexing to Fortran's 1-
based indexing for the triangulation module.

Created: 2026-08-18, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np

from formosa.geomorphology.meshing import triangulation as tri_m
from formosa.geomorphology._native import meshing_cstr_triangulation as cstrtri_f


def test_find_existing_constraints_fortran_classifies_mesh_edges():
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    nabrs = tri_m.find_facet_neighbours(faces, backend="fortran")
    constraints = np.array(
        [
            [0, 1],  # boundary edge
            [2, 1],  # reversed interior edge
            [3, 2],  # boundary edge owned by the second facet
            [0, 3],  # absent diagonal
            [2, 0],  # reversed boundary edge
        ],
        dtype=np.int32,
    )

    present, err_code = cstrtri_f.find_existing_constraints(
        tri_m._to_fortran_indices(faces),
        tri_m._to_fortran_neighbours(nabrs),
        4,
        tri_m._to_fortran_indices(constraints),
    )

    assert err_code == 0
    np.testing.assert_array_equal(present, [True, True, True, False, True])
