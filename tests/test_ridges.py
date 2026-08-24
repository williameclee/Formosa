"""
Verifies ridge-network parity between the Python and Fortran
backends.

Last modified: 2026-08-23, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np
import pytest

import formosa.geomorphology.drainage.ridges as ridges_m
from formosa import D8Directions
from formosa.utils import BACKENDS, Backend
from tests.test_metrics_parity import unequal_tributary_network


@pytest.mark.parametrize("backend", BACKENDS)
def test_ridge_strahler_order_forwards_valid_mask(
    unequal_tributary_network, backend: Backend
):
    dirs, valids, expected = unequal_tributary_network
    dir_scheme = D8Directions(transform_codes=lambda x: x)

    orders = ridges_m.compute_ridge_strahler_order(
        dirs, dir_scheme=dir_scheme, valids=valids, backend=backend, dir_is_ridge=True
    )

    np.testing.assert_array_equal(orders, expected)
