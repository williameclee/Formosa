"""
Tests shared utility routines in the FORTRAN backend.

Last modified: 2026-08-10, En-Chi Lee (williameclee@gmail.com)
"""

import pytest
from tests.core import *

import numpy as np

from formosa import D8Directions
import formosa.geomorphology.drainage.neighbours as utils_m
from formosa.geomorphology._native import utils as utils_f


@pytest.mark.parametrize(
    ("i", "j", "nrows", "ncols", "id"),
    [
        (1, 1, 3, 4, 1),
        (3, 4, 3, 4, 12),
        (0, 1, 3, 4, 0),
        (4, 1, 3, 4, 0),
        (1, 0, 3, 4, 0),
        (1, 5, 3, -2, 0),
        (1, 5, 3, 4, 0),
    ],
)
def test_ij2id(i, j, nrows, ncols, id):
    assert utils_f.ij2id_checked(i, j, nrows, ncols) == id


@pytest.mark.parametrize(
    ("id", "nrows", "ncols", "i", "j", "is_valid"),
    [
        (1, 3, 4, 1, 1, True),
        (12, 3, 4, 3, 4, True),
        (4, -2, 4, 0, 0, False),
        (4, 0, 4, 0, 0, False),
        (0, 3, 4, 0, 0, False),
        (13, 3, 4, 0, 0, False),
    ],
)
def test_id2ij(id, nrows, ncols, i, j, is_valid):
    assert utils_f.id2ij_checked(id, nrows, ncols) == (i, j, is_valid)


@pytest.mark.parametrize(
    ("mask", "max_len", "exp_cnt", "exp_err_code", "exp_ids"),
    argvalues=[
        (np.zeros((2, 2)), 4, 0, 0, None),
        (np.ones((2, 2)), 4, 4, 0, np.array([1, 2, 3, 4])),
        (
            np.array([[T, F, F], [F, T, T], [T, F, T]]),
            6,
            5,
            0,
            np.array([1, 3, 5, 8, 9]),
        ),
        (
            np.array([[F, F, T, F], [T, F, F, T], [T, T, T, T]]),
            7,
            7,
            0,
            np.array([2, 3, 6, 7, 9, 11, 12]),
        ),
    ],
)
def test_mask2id(mask, max_len, exp_cnt, exp_err_code, exp_ids):
    ids, cnt, err_code = utils_f.mask2id(mask.astype(bool, order="F"), max_len)

    assert cnt == exp_cnt
    assert err_code == exp_err_code
    if cnt > 0:
        np.testing.assert_array_equal(ids[:cnt], exp_ids[:cnt])


@pytest.mark.parametrize(
    ("mask", "max_len", "exp_cnt", "exp_err_code", "exp_ijs"),
    argvalues=[
        (np.zeros((2, 2)), 4, 0, 0, None),
        (np.ones((2, 2)), 4, 4, 0, np.array([[1, 1], [2, 1], [1, 2], [2, 2]]).T),
        (
            np.array([[T, F, F], [F, T, T], [T, F, T]]),
            6,
            5,
            0,
            np.array([[1, 1], [3, 1], [2, 2], [2, 3], [3, 3]]).T,
        ),
        (
            np.array([[F, F, T, F], [T, F, F, T], [T, T, T, T]]),
            7,
            7,
            0,
            np.array([[2, 1], [3, 1], [3, 2], [1, 3], [3, 3], [2, 4], [3, 4]]).T,
        ),
    ],
)
def test_mask2ij(mask, max_len, exp_cnt, exp_err_code, exp_ijs):
    ijs, cnt, err_code = utils_f.mask2ij(mask.astype(bool, order="F"), max_len)

    assert cnt == exp_cnt
    assert err_code == exp_err_code
    if cnt > 0:
        np.testing.assert_array_equal(ijs[:, :cnt], exp_ijs[:, :cnt])


def test_mask2id_mask2ij_overflow_error():
    _, cnt, err_code = utils_f.mask2id(np.ones((2, 2), dtype=bool, order="F"), 2)
    assert cnt == 2
    assert err_code == 3

    _, cnt, err_code = utils_f.mask2ij(np.ones((2, 2), dtype=bool, order="F"), 2)
    assert cnt == 2
    assert err_code == 3


def test_direction_utilities_infer_input_shapes():
    offsets = np.array([[0, 0], [0, 1], [0, -1]], dtype=np.int32, order="F")
    codes = np.array([0, 1, 5], dtype=np.int8)

    assert utils_f.find_noflow_code(offsets, codes) == 0
    assert np.array_equal(utils_f.find_opposite_codes(offsets, codes), [0, 5, 1])

    lookup = utils_f.fill_offset_lookup(offsets, codes)
    assert np.array_equal(lookup[1], [0, 1])
    assert np.array_equal(lookup[5], [0, -1])


def _assert_min_heap(queue, queue_size, elevations):
    for position in range(queue_size):
        left = 2 * position + 1
        right = left + 1
        parent_elevation = elevations[queue[position] - 1]
        if left < queue_size:
            assert parent_elevation <= elevations[queue[left] - 1]
        if right < queue_size:
            assert parent_elevation <= elevations[queue[right] - 1]


def _drain_priority_queue(queue, queue_size, elevations):
    popped_ids = []
    while queue_size.item() > 0:
        popped, err_code = utils_f.pop_priority_queue(queue, queue_size, elevations)
        assert err_code == 0
        popped_ids.append(popped)
        _assert_min_heap(queue, queue_size.item(), elevations)
    return popped_ids


@pytest.mark.parametrize(
    ("z", "exp_ids"),
    [
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
        ([9, 8, 7, 6, 5, 4], [6, 5, 4, 3, 2, 1]),
        ([5, 1, 4, 3, 2], [2, 5, 4, 3, 1]),
    ],
)
def test_priority_queue_push_pop_order(z, exp_ids):
    z = np.array(z, dtype=np.float32)
    queue = np.zeros(z.size, dtype=np.int32)
    queue_size = np.array(0, dtype=np.int32)

    for cell_id in range(1, z.size + 1):
        err_code = utils_f.push_priority_queue(queue, queue_size, cell_id, z)
        assert err_code == 0
        _assert_min_heap(queue, queue_size.item(), z)

    assert _drain_priority_queue(queue, queue_size, z) == exp_ids


def test_priority_queue_matches_sorted_random_elevations():
    rng = np.random.default_rng(20260806)
    z = rng.permutation(257).astype(np.float32)
    insertion_order = rng.permutation(z.size) + 1
    queue = np.zeros(z.size, dtype=np.int32)
    queue_size = np.array(0, dtype=np.int32)

    for cell_id in insertion_order:
        err_code = utils_f.push_priority_queue(queue, queue_size, int(cell_id), z)
        assert err_code == 0

    expected = (np.argsort(z) + 1).tolist()
    assert _drain_priority_queue(queue, queue_size, z) == expected


def test_priority_queue_handles_equal_elevations_and_reports_invalid_operations():
    z = np.array([2, 1, 1, 3], dtype=np.float32)
    queue = np.zeros(z.size, dtype=np.int32)
    queue_size = np.array(0, dtype=np.int32)

    for cell_id in range(1, z.size + 1):
        assert utils_f.push_priority_queue(queue, queue_size, cell_id, z) == 0

    popped_ids = _drain_priority_queue(queue, queue_size, z)
    popped_elevations = z[np.asarray(popped_ids) - 1]
    assert np.all(popped_elevations[:-1] <= popped_elevations[1:])

    popped, err_code = utils_f.pop_priority_queue(queue, queue_size, z)
    assert popped == 0
    assert err_code == 1

    one_cell_queue = np.zeros(1, dtype=np.int32)
    one_cell_size = np.array(0, dtype=np.int32)
    assert utils_f.push_priority_queue(one_cell_queue, one_cell_size, 1, z) == 0
    assert utils_f.push_priority_queue(one_cell_queue, one_cell_size, 2, z) == 3
    assert one_cell_size.item() == 1
    assert one_cell_queue[0] == 1

    invalid_size = np.array(-1, dtype=np.int32)
    assert utils_f.push_priority_queue(one_cell_queue, invalid_size, 1, z) == 1


@pytest.mark.parametrize(
    (
        *("dirs", "dir_scheme", "valids"),
        *("exp_dsi", "exp_dsj", "exp_dsij", "exp_inbounds"),
        "should_warn",
    ),
    [
        (
            [[3, 3, 3], [3, 3, 3], [1, 1, 0]],
            D8Directions(transform_codes=lambda x: x),
            None,
            [[1, 1, 1], [2, 2, 2], [2, 2, 2]],
            [[0, 1, 2], [0, 1, 2], [1, 2, 2]],
            [[1, 4, 7], [2, 5, 8], [5, 8, 8]],
            [[T, T, T], [T, T, T], [T, T, T]],
            False,
        ),
        (
            [[5, 1, 1], [5, 1, 1], [5, 1, 1]],
            D8Directions(transform_codes=lambda x: x),
            None,
            [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
            [[-1, 2, 3], [-1, 2, 3], [-1, 2, 3]],
            [[-3, 6, 9], [-2, 7, 10], [-1, 8, 11]],
            [[F, T, F], [F, T, F], [F, T, F]],
            True,
        ),
        (
            [[3, 3, 3], [3, 3, 3], [1, 1, 0]],
            D8Directions(transform_codes=lambda x: x),
            [[F, T, T], [T, T, T], [T, T, T]],
            [[-1, 1, 1], [2, 2, 2], [2, 2, 2]],
            [[-1, 1, 2], [0, 1, 2], [1, 2, 2]],
            [[-1, 4, 7], [2, 5, 8], [5, 8, 8]],
            [[T, T, T], [T, T, T], [T, T, T]],
            False,
        ),
        (
            [[1, 2, 2, 2], [8, 1, 1, 1], [8, 8, 8, 8], [1, 2, 1, 2]],
            D8Directions(transform_codes=lambda x: x),
            None,
            [[0, 1, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1], [3, 4, 3, 4]],
            [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
            None,
            [[T, T, T, F], [T, T, T, F], [T, T, T, F], [T, F, T, F]],
            True,
        ),
    ],
)
def test_downstreamid(
    dirs, dir_scheme, valids, exp_dsi, exp_dsj, exp_dsij, exp_inbounds, should_warn
):
    if should_warn:
        with pytest.raises(ValueError):
            dsi, dsj, dsij, ds_inbounds = utils_m.compute_downstream_indices(
                np.array(dirs),
                dir_scheme=dir_scheme,
                valids=np.array(valids) if valids is not None else None,
            )
        with pytest.warns(UserWarning):
            dsi, dsj, dsij, ds_inbounds = utils_m.compute_downstream_indices(
                np.array(dirs),
                dir_scheme=dir_scheme,
                valids=np.array(valids) if valids is not None else None,
                check=False,
            )
    else:
        dsi, dsj, dsij, ds_inbounds = utils_m.compute_downstream_indices(
            np.array(dirs),
            dir_scheme=dir_scheme,
            valids=np.array(valids) if valids is not None else None,
        )
    np.testing.assert_array_equal(dsi, np.array(exp_dsi))
    np.testing.assert_array_equal(dsj, np.array(exp_dsj))
    if exp_dsij is not None:
        np.testing.assert_array_equal(dsij, np.array(exp_dsij))
    np.testing.assert_array_equal(ds_inbounds, np.array(exp_inbounds))
