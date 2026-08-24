"""
Tests direction encoding conversions and custom scheme workflows.

Created: 2026-08-24, En-Chi Lee (williameclee@gmail.com)
Last modified: 2026-08-24, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np

from formosa import D8DirectionEncoding
from formosa.geomorphology.drainage import metrics


def test_code_to_offset_converts_an_array_without_collapsing_its_shape():
    dir_enc = D8DirectionEncoding(code_trans_func=lambda code: code)
    codes = np.array([[1, 2, 3], [4, 5, 0]], dtype=np.uint8)

    di, dj = dir_enc.code_to_offset(codes)

    np.testing.assert_array_equal(di, [[0, 1, 1], [1, 0, 0]])
    np.testing.assert_array_equal(dj, [[1, 1, 0], [-1, -1, 0]])
    assert di.shape == codes.shape
    assert dj.shape == codes.shape


def test_strahler_order_can_process_multiple_source_cells():
    dir_enc = D8DirectionEncoding(code_trans_func=lambda code: code)
    dirs = np.array([[3, 3, 3], [3, 3, 3], [1, 1, 0]], dtype=np.uint8)

    orders = metrics.compute_flow_strahler_order(dirs, dir_enc, backend="python")

    np.testing.assert_array_equal(orders, [[1, 1, 1], [1, 1, 1], [1, 2, 2]])
