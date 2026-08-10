"""
Tests flow-graph topology validation across configured backends.

Last modified: 2026-08-10, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np
import pytest

from formosa.utils import BACKENDS
import formosa.geomorphology.drainage.network.validation as val_m


@pytest.mark.parametrize("backend", BACKENDS)
def test_locate_invalid_graph_topology(backend):
    vtxs = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
    endpts = np.array([[0, 1], [2, 3]])
    expected = np.array([[0, 1, 0, 2, 1]], dtype=np.int32)
    np.testing.assert_array_equal(
        val_m.locate_invalid_graph_topology(vtxs, endpts, backend=backend),
        expected,
    )

    vtxs = np.array([[0, 0], [1, 1], [2, 0], [1, 0], [0, 1]])
    endpts = np.array([[0, 2], [3, 4]])
    expected = np.array([[0, 1, 0, 3, 1]], dtype=np.int32)
    np.testing.assert_array_equal(
        val_m.locate_invalid_graph_topology(vtxs, endpts, backend=backend),
        expected,
    )

    vtxs = np.array([[0, 0], [2, 2], [2, 0], [0, 2]])
    endpts = np.array([[0, 3]])
    expected = np.array([[0, 0, 0, 2, 1]], dtype=np.int32)
    np.testing.assert_array_equal(
        val_m.locate_invalid_graph_topology(vtxs, endpts, backend=backend),
        expected,
    )

    vtxs = np.array([[0, 0], [1, 1], [2, 2]])
    endpts = np.array([[0, 2]])
    assert val_m.locate_invalid_graph_topology(vtxs, endpts, backend=backend) is None

    with pytest.raises(ValueError, match="Invalid array shapes passed"):
        val_m.locate_invalid_graph_topology(
            np.array([1, 2, 3]), endpts, backend=backend
        )
