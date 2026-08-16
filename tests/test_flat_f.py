"""
Tests flat resolution using the FORTRAN backend.

This module covers native results, boundary cases, and translation
of FORTRAN status codes by the public drainage API.

Last modified: 2026-08-10, En-Chi Lee (williameclee@gmail.com)
"""

import pytest
import numpy as np

from formosa import D8Directions
import formosa.geomorphology.drainage.flat_resolution as flat_m
from formosa.geomorphology._native import drainage_flat_resolution as flat_f

from types import SimpleNamespace


@pytest.mark.parametrize(
    ("err_code", "exception", "detail"),
    [
        (1, ValueError, "invalid input"),
        (2, MemoryError, "allocate backend workspace"),
        (3, RuntimeError, "exceeded some array or index capacity"),
        (99, RuntimeError, "unknown"),
    ],
)
def test_label_flats_translates_fortran_errors(
    monkeypatch: pytest.MonkeyPatch, err_code, exception, detail
):
    def fake_label(*args):
        return np.zeros((1, 1), dtype=np.int32), err_code

    monkeypatch.setattr(flat_m, "flat_f", SimpleNamespace(label_flats=fake_label))

    with pytest.raises(exception, match=rf"label_flats.*{detail}.*{err_code}"):
        flat_m.label_flats(
            np.zeros((1, 1), dtype=np.float32), np.ones((1, 1), dtype=bool)
        )


def test_flat_synthetic_gradients_follow_breadth_first_layers():
    labels = np.ones((5, 5), dtype=np.int32, order="F")
    offsets = D8Directions().offsets.astype(np.int32, order="F")
    centre = np.zeros(labels.shape, dtype=bool, order="F")
    centre[2, 2] = True

    pulling, err_code = flat_f.create_pulling_syn_grad(labels, centre, offsets)
    assert err_code == 0
    np.testing.assert_array_equal(
        pulling,
        np.array(
            [
                [3, 3, 3, 3, 3],
                [3, 2, 2, 2, 3],
                [3, 2, 1, 2, 3],
                [3, 2, 2, 2, 3],
                [3, 3, 3, 3, 3],
            ],
            dtype=np.int32,
        ),
    )

    pushing, err_code = flat_f.create_pushing_syn_grad(labels, centre, offsets)
    assert err_code == 0
    np.testing.assert_array_equal(pushing, 4 - pulling)


def test_flat_synthetic_gradients_handle_empty_inputs():
    labels = np.zeros((2, 3), dtype=np.int32, order="F")
    edges = np.zeros(labels.shape, dtype=bool, order="F")
    offsets = D8Directions().offsets.astype(np.int32, order="F")

    pushing, pushing_err = flat_f.create_pushing_syn_grad(labels, edges, offsets)
    pulling, pulling_err = flat_f.create_pulling_syn_grad(labels, edges, offsets)

    assert pushing_err == 0
    assert pulling_err == 0
    np.testing.assert_array_equal(pushing, 0)
    np.testing.assert_array_equal(pulling, 0)
