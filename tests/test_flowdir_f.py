# Last modified,
#   2026-08-03, En-Chi Lee (williameclee@gmail.com)
#     - Added test cases for function `find_acyclic_flowdirs`.
#   2026-08-04, En-Chi Lee (williameclee@gmail.com)
#     - Added test cases for FORTRAN error code handling.

import pytest
import numpy as np

from formosa import D8Directions
import formosa.geomorphology.drainage.flowdir as flowdir_m
from formosa.geomorphology.drainage_f import drainage_flowdir as drainage_f

from types import SimpleNamespace


@pytest.mark.parametrize(
    ("err_code", "exception", "detail"),
    [
        (1, ValueError, "invalid input"),
        (2, MemoryError, "allocate backend workspace"),
        (3, RuntimeError, "array or index capacity exceeded"),
        (99, RuntimeError, "unknown"),
    ],
)
def test_label_flats_translates_fortran_errors(
    monkeypatch: pytest.MonkeyPatch, err_code, exception, detail
):
    def fake_label(*args):
        return np.zeros((1, 1), dtype=np.int32), err_code

    monkeypatch.setattr(flowdir_m, "drainage_f", SimpleNamespace(label_flats=fake_label))

    with pytest.raises(exception, match=rf"label_flats.*{detail}.*{err_code}"):
        flowdir_m.label_flats(
            np.zeros((1, 1), dtype=np.float32), np.ones((1, 1), dtype=bool)
        )


def test_flat_synthetic_gradients_follow_breadth_first_layers():
    labels = np.ones((5, 5), dtype=np.int32, order="F")
    offsets = D8Directions().offsets.astype(np.int32, order="F")
    centre = np.zeros(labels.shape, dtype=bool, order="F")
    centre[2, 2] = True

    pulling, err_code = drainage_f.create_pulling_syn_grad(labels, centre, offsets)
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

    pushing, err_code = drainage_f.create_pushing_syn_grad(labels, centre, offsets)
    assert err_code == 0
    np.testing.assert_array_equal(pushing, 4 - pulling)


def test_flat_synthetic_gradients_handle_empty_inputs():
    labels = np.zeros((2, 3), dtype=np.int32, order="F")
    edges = np.zeros(labels.shape, dtype=bool, order="F")
    offsets = D8Directions().offsets.astype(np.int32, order="F")

    pushing, pushing_err = drainage_f.create_pushing_syn_grad(labels, edges, offsets)
    pulling, pulling_err = drainage_f.create_pulling_syn_grad(labels, edges, offsets)

    assert pushing_err == 0
    assert pulling_err == 0
    np.testing.assert_array_equal(pushing, 0)
    np.testing.assert_array_equal(pulling, 0)


def test_indegree_3x3():
    dir_scheme = D8Directions(transform_codes=lambda x: x)

    # Config 1
    dirs = np.array([[3, 3, 3], [3, 3, 3], [1, 1, 0]])

    expected_indegs = np.array(
        [
            [0, 0, 0],
            [1, 1, 1],
            [1, 2, 2],
        ]
    )

    np.testing.assert_array_equal(
        flowdir_m.count_indegree(dirs, dir_scheme=dir_scheme, backend="fortran"),
        expected_indegs,
    )

    # Config 2
    dirs = np.array([[5, 1, 1], [5, 1, 1], [5, 1, 1]])

    expected_indegs = np.array(
        [
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
        ]
    )

    np.testing.assert_array_equal(
        flowdir_m.count_indegree(dirs, dir_scheme=dir_scheme, backend="fortran"),
        expected_indegs,
    )


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
        flowdir_m,
        "drainage_f",
        SimpleNamespace(find_acyclic_flowdirs=fake_find),
    )

    with pytest.raises(exception):
        flowdir_m.find_acyclic_flowdirs(
            np.zeros((1, 1), dtype=np.uint8),
            indegs=np.zeros((1, 1), dtype=np.int8),
            backend="fortran",
        )
