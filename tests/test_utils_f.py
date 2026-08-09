"""
Tests related to FORTRAN backend utility functions.

Last modified: 2026-08-09, En-Chi Lee (williameclee@gmail.com)
"""

from tests.core import *

import pytest
import numpy as np

from formosa import D8Directions
import formosa.geomorphology.drainage.neighbours as utils_m
from formosa.geomorphology._native import utils as utils_f


def test_checked_linear_cell_ids_reject_invalid_coordinates_and_ids():
    assert utils_f.ij2id_checked(1, 1, 3, 4) == 1
    assert utils_f.ij2id_checked(3, 4, 3, 4) == 12
    assert utils_f.ij2id_checked(0, 1, 3, 4) == 0
    assert utils_f.ij2id_checked(4, 1, 3, 4) == 0
    assert utils_f.ij2id_checked(1, 0, 3, 4) == 0
    assert utils_f.ij2id_checked(1, 5, 3, 4) == 0

    assert utils_f.id2ij_checked(1, 3, 4) == (1, 1, True)
    assert utils_f.id2ij_checked(12, 3, 4) == (3, 4, True)
    assert utils_f.id2ij_checked(0, 3, 4) == (0, 0, False)
    assert utils_f.id2ij_checked(13, 3, 4) == (0, 0, False)


def test_mask2ij_returns_output_capacity_error():
    indices, count, err_code = utils_f.mask2ij(
        np.ones((2, 2), dtype=bool, order="F"),
        2,
    )

    assert count == 2
    assert err_code == 3
    assert indices.shape == (2, 2)


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


def test_priority_queue_pushes_into_empty_queue_and_pops_in_elevation_order():
    z = np.array([5, 1, 4, 3, 2], dtype=np.float32)
    queue = np.zeros(z.size, dtype=np.int32)
    queue_size = np.array(0, dtype=np.int32)

    for cell_id in range(1, z.size + 1):
        err_code = utils_f.push_priority_queue(queue, queue_size, cell_id, z)
        assert err_code == 0
        _assert_min_heap(queue, queue_size.item(), z)

    assert _drain_priority_queue(queue, queue_size, z) == [2, 5, 4, 3, 1]


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


def test_downstreamid_3x3():
    dir_scheme = D8Directions(transform_codes=lambda x: x)

    # Config 1
    dirs = np.array(
        [
            [3, 3, 3],
            [3, 3, 3],
            [1, 1, 0],
        ]
    )

    expected_dsi = np.array(
        [
            [1, 1, 1],
            [2, 2, 2],
            [2, 2, 2],
        ]
    )
    expected_dsj = np.array(
        [
            [0, 1, 2],
            [0, 1, 2],
            [1, 2, 2],
        ]
    )
    expected_dsij = np.array(
        [
            [1, 4, 7],
            [2, 5, 8],
            [5, 8, 8],
        ]
    )
    expected_inbounds = np.array(
        [
            [T, T, T],
            [T, T, T],
            [T, T, T],
        ]
    )

    dsi, dsj, dsij, ds_inbounds = utils_m.compute_downstream_indices(
        dirs, dir_scheme=dir_scheme
    )

    np.testing.assert_array_equal(dsi, expected_dsi)
    np.testing.assert_array_equal(dsj, expected_dsj)
    np.testing.assert_array_equal(dsij, expected_dsij)
    np.testing.assert_array_equal(ds_inbounds, expected_inbounds)

    # Config 2
    dirs = np.array(
        [
            [5, 1, 1],
            [5, 1, 1],
            [5, 1, 1],
        ]
    )

    expected_dsi = np.array(
        [
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
        ]
    )
    expected_dsj = np.array(
        [
            [-1, 2, 3],
            [-1, 2, 3],
            [-1, 2, 3],
        ]
    )
    expected_dsij = np.array(
        [
            [-3, 6, 9],
            [-2, 7, 10],
            [-1, 8, 11],
        ]
    )
    expected_inbounds = np.array(
        [
            [F, T, F],
            [F, T, F],
            [F, T, F],
        ]
    )

    with pytest.warns(UserWarning):
        dsi, dsj, dsij, ds_inbounds = utils_m.compute_downstream_indices(
            dirs, dir_scheme=dir_scheme, check=False
        )

    np.testing.assert_array_equal(dsi, expected_dsi)
    np.testing.assert_array_equal(dsj, expected_dsj)
    np.testing.assert_array_equal(dsij, expected_dsij)
    np.testing.assert_array_equal(ds_inbounds, expected_inbounds)

    # Config 4 - with validity mask
    dirs = np.array(
        [
            [3, 3, 3],
            [3, 3, 3],
            [1, 1, 0],
        ]
    )
    valids = np.array([[F, T, T], [T, T, T], [T, T, T]])

    expected_dsi = np.array(
        [
            [-1, 1, 1],
            [2, 2, 2],
            [2, 2, 2],
        ]
    )
    expected_dsj = np.array(
        [
            [-1, 1, 2],
            [0, 1, 2],
            [1, 2, 2],
        ]
    )
    expected_dsij = np.array(
        [
            [-1, 4, 7],
            [2, 5, 8],
            [5, 8, 8],
        ]
    )
    expected_inbounds = np.array(
        [
            [T, T, T],
            [T, T, T],
            [T, T, T],
        ]
    )

    dsi, dsj, dsij, ds_inbounds = utils_m.compute_downstream_indices(
        dirs, dir_scheme=dir_scheme, valids=valids
    )

    np.testing.assert_array_equal(dsi, expected_dsi)
    np.testing.assert_array_equal(dsj, expected_dsj)
    np.testing.assert_array_equal(dsij, expected_dsij)
    np.testing.assert_array_equal(ds_inbounds, expected_inbounds)


def test_downstreamid_4x4():
    dir_scheme = D8Directions(transform_codes=lambda x: x)

    # Config 1
    dirs = np.array(
        [
            [1, 2, 2, 2],
            [8, 1, 1, 1],
            [8, 8, 8, 8],
            [1, 2, 1, 2],
        ]
    )
    expected_dsi = np.array(
        [
            [0, 1, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1],
            [3, 4, 3, 4],
        ]
    )
    expected_dsj = np.array(
        [
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
        ]
    )
    expected_valids = np.array(
        [
            [T, T, T, F],
            [T, T, T, F],
            [T, T, T, F],
            [T, F, T, F],
        ]
    )
    with pytest.warns(UserWarning):
        dsi, dsj, _, ds_valids = utils_m.compute_downstream_indices(
            dirs, dir_scheme=dir_scheme, check=F
        )
    np.testing.assert_array_equal(dsi, expected_dsi)
    np.testing.assert_array_equal(dsj, expected_dsj)
    np.testing.assert_array_equal(ds_valids, expected_valids)
