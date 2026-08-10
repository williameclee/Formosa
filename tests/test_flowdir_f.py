"""
Tests flow-direction derivation using the FORTRAN backend.

Last modified: 2026-08-10, En-Chi Lee (williameclee@gmail.com)
"""

import pytest
import numpy as np

import formosa.geomorphology.drainage.flowdir as flowdir_m

from types import SimpleNamespace


@pytest.mark.parametrize(
    ("err_code", "exception"),
    [
        (1, ValueError),
        (2, MemoryError),
        (3, RuntimeError),
        (99, RuntimeError),
    ],
)
def test_find_acyclic_flowdirs_translates_fortran_errors(
    monkeypatch: pytest.MonkeyPatch, err_code, exception
):
    def fake_find(*args):
        return np.zeros((1, 1), dtype=bool), err_code

    monkeypatch.setattr(
        flowdir_m, "flowdir_f", SimpleNamespace(find_acyclic_flowdirs=fake_find)
    )

    with pytest.raises(exception):
        flowdir_m.find_acyclic_flowdirs(
            np.zeros((1, 1), dtype=np.uint8),
            indegs=np.zeros((1, 1), dtype=np.int8),
            backend="fortran",
        )
