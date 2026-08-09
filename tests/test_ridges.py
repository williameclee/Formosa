"""
Parity tests related to construction of the ridge network using the
2 backends.

Last modified: 2026-08-09, En-Chi Lee (williameclee@gmail.com)
"""

import pytest
import numpy as np

from formosa.utils import BACKENDS
from formosa import D8Directions
import formosa.geomorphology.drainage.ridges as ridges_m
from tests.test_metrics_parity import unequal_tributary_network


@pytest.mark.parametrize("backend", BACKENDS)
def test_ridge_strahler_order_forwards_valid_mask(unequal_tributary_network, backend):
    dirs, valids, expected = unequal_tributary_network
    dir_scheme = D8Directions(transform_codes=lambda x: x)

    orders = ridges_m.compute_ridge_strahler_order(
        dirs, dir_scheme=dir_scheme, valids=valids, backend=backend, dir_is_ridge=True
    )

    np.testing.assert_array_equal(orders, expected)
