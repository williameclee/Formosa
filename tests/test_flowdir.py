"""
Parity tests related to the derivation of flow directions using the
2 backends.

Last modified: 2026-08-09, En-Chi Lee (williameclee@gmail.com)
"""

from tests.core import *

import pytest
import numpy as np


from formosa.utils import BACKENDS
from formosa import D8Directions
import formosa.geomorphology.drainage.flowdir as flowdir_m


@pytest.mark.parametrize("backend", BACKENDS)
def test_find_flowdir_cycles_with_feeder_and_invalid_cell(backend):
    dir_scheme = D8Directions(transform_codes=lambda x: x)
    # Cell 0 feeds the cycle between cells 1 and 2; cell 3 is invalid.
    dirs = np.array([[1, 1, 5, 0]], dtype=np.uint8)
    valids = np.array([[T, T, T, F]])

    acyclics = flowdir_m.find_acyclic_flowdirs(
        dirs, dir_scheme=dir_scheme, valids=valids, backend=backend
    )
    cyclics = flowdir_m.find_cyclic_flowdirs(
        dirs, dir_scheme=dir_scheme, valids=valids, backend=backend
    )

    np.testing.assert_array_equal(acyclics, [[T, F, F, F]])
    np.testing.assert_array_equal(cyclics, [[F, T, T, F]])


@pytest.mark.parametrize("backend", BACKENDS)
def test_find_flowdir_cycles_accepts_supplied_indegrees(backend):
    dir_scheme = D8Directions(transform_codes=lambda x: x)
    dirs = np.array([[1, 1, 0]], dtype=np.uint8)
    valids = np.ones(dirs.shape, dtype=bool)
    indegs = np.array([[0, 1, 1]], dtype=np.int8)
    original_indegs = indegs.copy()

    acyclics = flowdir_m.find_acyclic_flowdirs(
        dirs,
        dir_scheme=dir_scheme,
        valids=valids,
        indegs=indegs,
        backend=backend,
    )

    np.testing.assert_array_equal(acyclics, valids)
    np.testing.assert_array_equal(indegs, original_indegs)


def test_find_acyclic_flowdirs_default_code_128_backend_parity():
    dirs = np.array([[0, 0], [128, 0]], dtype=np.uint8)
    valids = np.array([[F, T], [T, F]])

    python_acyclics = flowdir_m.find_acyclic_flowdirs(
        dirs, valids=valids, backend="python"
    )
    fortran_acyclics = flowdir_m.find_acyclic_flowdirs(
        dirs, valids=valids, backend="fortran"
    )

    np.testing.assert_array_equal(fortran_acyclics, python_acyclics)
    np.testing.assert_array_equal(fortran_acyclics, valids)
