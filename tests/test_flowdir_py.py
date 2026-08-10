"""
Tests flow-direction derivation using the Python backend.

Last modified: 2026-08-10, En-Chi Lee (williameclee@gmail.com)
"""

import pytest
import numpy as np

import formosa.geomorphology.drainage.flowdir as flowdir_m


@pytest.mark.parametrize("name", ("valids", "indegs"))
def test_find_acyclic_flowdirs_rejects_shape_mismatch(name):
    kwargs = {name: np.ones((2, 1), dtype=bool)}
    with pytest.raises(ValueError, match="Shapes"):
        flowdir_m.find_acyclic_flowdirs(
            np.zeros((1, 1), dtype=np.uint8), backend="python", **kwargs  # type: ignore
        )
