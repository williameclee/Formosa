"""
Tests flow-direction derivation using the Fortran backend.

Last modified: 2026-08-24, En-Chi Lee (williameclee@gmail.com)
"""

from types import SimpleNamespace

import numpy as np
import pytest

import formosa.geomorphology.drainage.flowdir as flowdir_m


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
    monkeypatch: pytest.MonkeyPatch, err_code: int, exception: Exception
):
    def fake_find(*args):
        return np.zeros((1, 1), dtype=bool), err_code

    monkeypatch.setattr(
        flowdir_m, "flowdir_f", SimpleNamespace(find_acyclic_flowdirs=fake_find)
    )

    with pytest.raises(exception):  # pyright: ignore[reportArgumentType]
        _ = flowdir_m.find_acyclic_flowdirs(
            np.zeros((1, 1), dtype=np.uint8),
            indegs=np.zeros((1, 1), dtype=np.int8),
            backend="fortran",
        )
