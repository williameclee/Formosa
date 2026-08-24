"""
Tests flow-direction derivation using the Python backend.

Last modified: 2026-08-24, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np
import pytest

import formosa.geomorphology.drainage.flowdir as flowdir_m


@pytest.mark.parametrize("name", ("valids", "indegs"))
def test_find_acyclic_flowdirs_rejects_shape_mismatch(name: str):
    kwargs = {name: np.ones((2, 1), dtype=bool)}
    with pytest.raises(ValueError, match="Shapes"):
        _ = flowdir_m.find_acyclic_flowdirs(
            np.zeros((1, 1), dtype=np.uint8),
            backend="python",
            **kwargs,
        )
