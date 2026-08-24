"""
Verifies ridge-network parity between the Python and Fortran
backends.

Last modified: 2026-08-24, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np
import pytest

import formosa.geomorphology.drainage.ridges as ridges_m
from formosa import D8DirectionEncoding
from formosa.utils import BACKENDS, Backend
from tests.test_metrics_parity import unequal_tributary_network


@pytest.mark.parametrize("backend", BACKENDS)
def test_ridge_strahler_order_forwards_valid_mask(
    unequal_tributary_network, backend: Backend
):
    dirs, valids, exp = unequal_tributary_network
    dir_enc = D8DirectionEncoding(code_trans_func=lambda x: x)

    orders = ridges_m.compute_ridge_strahler_order(
        dirs, dir_enc, valids=valids, dir_is_ridge=True, backend=backend
    )

    np.testing.assert_array_equal(orders, exp)
