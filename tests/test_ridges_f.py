"""
Tests ridge-network construction using the FORTRAN backend.

Last modified: 2026-08-10, En-Chi Lee (williameclee@gmail.com)
"""

import pytest
import numpy as np

from formosa import D8Directions
import formosa.geomorphology.drainage.ridges as ridges_m
from formosa.geomorphology._native import drainage_ridges as ridges_f

from types import SimpleNamespace


def _reference_max_branch_dist(dirs, valids, x, y, dir_scheme):
    """Small direct path-tracing reference for the bulk backend."""
    offsets = dir_scheme.offset_dict
    nrows, ncols = dirs.shape

    def path_from(i, j):
        path = [(i, j)]
        cumulative = [0.0]
        seen = {(i, j)}
        while True:
            di, dj = offsets.get(int(dirs[i, j]), (0, 0))
            ni, nj = i + di, j + dj
            if (ni, nj) == (i, j):
                break
            if not (0 <= ni < nrows and 0 <= nj < ncols and valids[ni, nj]):
                break
            if (ni, nj) in seen:
                raise ValueError("cycle")
            cumulative.append(
                cumulative[-1] + np.hypot(x[ni, nj] - x[i, j], y[ni, nj] - y[i, j])
            )
            path.append((ni, nj))
            seen.add((ni, nj))
            i, j = ni, nj
        return path, np.asarray(cumulative)

    paths = {
        (int(i), int(j)): path_from(int(i), int(j)) for i, j in np.argwhere(valids)
    }
    result = np.zeros(dirs.shape, dtype=np.float32)
    for (i, j), (path1, dist1) in paths.items():
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if (di == 0 and dj == 0) or (i + di, j + dj) not in paths:
                    continue
                path2, _ = paths[(i + di, j + dj)]
                path2_ids = {cell: index for index, cell in enumerate(path2)}
                confluence_index = next(
                    (index for index, cell in enumerate(path1) if cell in path2_ids),
                    None,
                )
                branch_dist = (
                    dist1[-1] if confluence_index is None else dist1[confluence_index]
                )
                result[i, j] = max(result[i, j], branch_dist)
    return result


def test_max_branch_distance_matches_direct_path_reference():
    rng = np.random.default_rng(2817)
    dir_scheme = D8Directions(transform_codes=lambda value: value)
    code_by_offset = {
        tuple(offset): int(code)
        for code, offset in zip(dir_scheme.codes, dir_scheme.offsets)
    }
    shape = (9, 11)
    dirs = np.zeros(shape, dtype=np.uint8, order="F")
    valids = rng.random(shape) > 0.12
    choices = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for i, j in np.ndindex(shape):
        candidates = [
            offset
            for offset in choices
            if i + offset[0] < shape[0] and j + offset[1] < shape[1]
        ]
        dirs[i, j] = code_by_offset[candidates[rng.integers(len(candidates))]]

    rows, cols = np.meshgrid(
        np.arange(shape[0], dtype=np.float32),
        np.arange(shape[1], dtype=np.float32),
        indexing="ij",
    )
    x = cols + rows * np.float32(0.17)
    y = rows + cols * np.float32(0.09)

    expected = _reference_max_branch_dist(dirs, valids, x, y, dir_scheme)
    actual = ridges_m.compute_dist2conf_max(
        dirs, valids=valids, x=x, y=y, dir_scheme=dir_scheme
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


def test_max_branch_distance_propagates_from_sink_to_non_sink_cells():
    dir_scheme = D8Directions(transform_codes=lambda value: value)
    dirs = np.array([[2, 3], [1, 0]], dtype=np.uint8, order="F")

    actual = ridges_m.compute_dist2conf_max(
        dirs,
        valids=np.ones(dirs.shape, dtype=bool, order="F"),
        dir_scheme=dir_scheme,
    )

    np.testing.assert_allclose(actual, [[np.sqrt(2), 1.0], [1.0, 0.0]])


def test_max_branch_distance_parallel_metadata_propagation():
    dir_scheme = D8Directions(transform_codes=lambda value: value)
    code_by_offset = {
        tuple(offset): int(code)
        for code, offset in zip(dir_scheme.codes, dir_scheme.offsets)
    }
    ncols = 32768
    dirs = np.full((2, ncols), code_by_offset[(0, 0)], dtype=np.uint8, order="F")
    dirs[0, :] = code_by_offset[(1, 0)]

    actual = ridges_m.compute_dist2conf_max(
        dirs, valids=np.ones(dirs.shape, dtype=bool, order="F"), dir_scheme=dir_scheme
    )

    np.testing.assert_array_equal(actual[0, :], 1.0)
    np.testing.assert_array_equal(actual[1, :], 0.0)


def test_max_branch_distance_reports_cycle():
    dir_scheme = D8Directions(transform_codes=lambda value: value)
    code_by_offset = {
        tuple(offset): int(code)
        for code, offset in zip(dir_scheme.codes, dir_scheme.offsets)
    }
    dirs = np.array(
        [[code_by_offset[(0, 1)], code_by_offset[(0, -1)]]],
        dtype=np.uint8,
        order="F",
    )
    valids = np.ones(dirs.shape, dtype=bool, order="F")
    x = np.array([[0.0, 1.0]], dtype=np.float32, order="F")
    y = np.zeros(dirs.shape, dtype=np.float32, order="F")

    _, err_code = ridges_f.compute_max_branch_dist(
        dirs,
        valids,
        x,
        y,
        dir_scheme.offsets.astype(np.int32, order="F"),
        dir_scheme.codes.astype(np.uint8, order="F"),
    )
    assert err_code == 1


def test_confluence_distance_2x2():
    dir_scheme = D8Directions(transform_codes=lambda x: x)
    offset_lookup = np.zeros((256, 2), dtype=np.int32)
    for code, (di, dj) in zip(dir_scheme.codes, dir_scheme.offsets):
        offset_lookup[code, :] = [di, dj]

    dirs = np.array([[3, 3], [1, 0]], dtype=np.uint8)
    x, y = np.meshgrid(
        np.arange(dirs.shape[1], dtype=np.float32),
        np.arange(dirs.shape[0], dtype=np.float32),
        indexing="xy",
    )

    common_kwargs = {
        "dirs": dirs.astype(np.uint8, order="F"),
        "x": x.astype(np.float32, order="F"),
        "y": y.astype(np.float32, order="F"),
        "offset_lookup": offset_lookup,
        "check_flag": True,
    }

    dists, err_code = ridges_f.compute_confluence_dist([1, 1], [1, 2], **common_kwargs)
    assert np.isclose(dists[0], 2.0)
    assert np.isclose(dists[1], 1.0)

    dists, err_code = ridges_f.compute_confluence_dist([2, 1], [1, 2], **common_kwargs)
    assert np.isclose(dists[0], 1.0)
    assert np.isclose(dists[1], 1.0)

    dists, err_code = ridges_f.compute_confluence_dist([1, 1], [2, 1], **common_kwargs)
    assert np.isclose(dists[0], 1.0)
    assert np.isclose(dists[1], 0.0)

    dists, err_code = ridges_f.compute_confluence_dist([1, 1], [1, 1], **common_kwargs)
    assert np.isclose(dists[0], 0.0)
    assert np.isclose(dists[1], 0.0)

    common_kwargs["dirs"] = np.array([[3, 3], [5, 1]], dtype=np.uint8, order="F")

    dists, err_code = ridges_f.compute_confluence_dist([1, 1], [1, 2], **common_kwargs)
    assert np.isclose(dists[0], 1.0)
    assert np.isclose(dists[1], 1.0)

    dists, err_code = ridges_f.compute_confluence_dist([2, 1], [1, 2], **common_kwargs)
    assert np.isclose(dists[0], 0.0)
    assert np.isclose(dists[1], 1.0)

    dists, err_code = ridges_f.compute_confluence_dist([2, 1], [2, 2], **common_kwargs)
    assert np.isclose(dists[0], 0.0)
    assert np.isclose(dists[1], 0.0)

    common_kwargs["dirs"] = np.array([[2, 3], [1, 0]], dtype=np.uint8, order="F")

    dists, err_code = ridges_f.compute_confluence_dist([1, 1], [1, 2], **common_kwargs)
    assert np.isclose(dists[0], np.sqrt(2))
    assert np.isclose(dists[1], 1.0)

    dists, err_code = ridges_f.compute_confluence_dist([1, 1], [2, 1], **common_kwargs)
    assert np.isclose(dists[0], np.sqrt(2))
    assert np.isclose(dists[1], 1.0)

    dists, err_code = ridges_f.compute_confluence_dist([1, 1], [2, 2], **common_kwargs)
    assert np.isclose(dists[0], np.sqrt(2))
    assert np.isclose(dists[1], 0.0)

    common_kwargs["dirs"] = np.array([[1, 0], [1, 7]], dtype=np.uint8, order="F")

    dists, err_code = ridges_f.compute_confluence_dist([1, 1], [1, 2], **common_kwargs)
    assert np.isclose(dists[0], 1.0)
    assert np.isclose(dists[1], 0.0)

    dists, err_code = ridges_f.compute_confluence_dist([1, 1], [2, 1], **common_kwargs)
    assert np.isclose(dists[0], 1.0)
    assert np.isclose(dists[1], 2.0)

    dists, err_code = ridges_f.compute_confluence_dist([2, 2], [2, 1], **common_kwargs)
    assert np.isclose(dists[0], 0.0)
    assert np.isclose(dists[1], 1.0)

    common_kwargs["dirs"] = np.array([[0, 5], [7, 7]], dtype=np.uint8, order="F")

    dists, err_code = ridges_f.compute_confluence_dist([1, 1], [1, 2], **common_kwargs)
    assert np.isclose(dists[0], 0.0)
    assert np.isclose(dists[1], 1.0)

    dists, err_code = ridges_f.compute_confluence_dist([1, 1], [2, 2], **common_kwargs)
    assert np.isclose(dists[0], 0.0)
    assert np.isclose(dists[1], 2.0)

    dists, err_code = ridges_f.compute_confluence_dist([2, 1], [2, 2], **common_kwargs)
    assert np.isclose(dists[0], 1.0)
    assert np.isclose(dists[1], 2.0)

    common_kwargs["dirs"] = np.array([[1, 0], [8, 7]], dtype=np.uint8, order="F")

    dists, err_code = ridges_f.compute_confluence_dist([1, 1], [1, 2], **common_kwargs)
    assert np.isclose(dists[0], 1.0)
    assert np.isclose(dists[1], 0.0)

    dists, err_code = ridges_f.compute_confluence_dist([1, 1], [2, 1], **common_kwargs)
    assert np.isclose(dists[0], 1.0)
    assert np.isclose(dists[1], np.sqrt(2))

    common_kwargs["dirs"] = np.array([[2, 3], [1, 0]], dtype=np.uint8, order="F")

    dists, err_code = ridges_f.compute_confluence_dist([1, 1], [1, 2], **common_kwargs)
    assert np.isclose(dists[0], np.sqrt(2))
    assert np.isclose(dists[1], 1.0)

    dists, err_code = ridges_f.compute_confluence_dist([1, 1], [2, 1], **common_kwargs)
    assert np.isclose(dists[0], np.sqrt(2))
    assert np.isclose(dists[1], 1.0)

    common_kwargs["dirs"] = np.array([[0, 5], [7, 6]], dtype=np.uint8, order="F")

    dists, err_code = ridges_f.compute_confluence_dist([1, 1], [1, 2], **common_kwargs)
    assert np.isclose(dists[0], 0)
    assert np.isclose(dists[1], 1.0)

    dists, err_code = ridges_f.compute_confluence_dist([1, 1], [2, 2], **common_kwargs)
    assert np.isclose(dists[0], 0.0)
    assert np.isclose(dists[1], np.sqrt(2))


def test_confluence_distance_3x3():
    dir_scheme = D8Directions(transform_codes=lambda x: x)
    offset_lookup = np.zeros((256, 2), dtype=np.int32)
    for code, (di, dj) in zip(dir_scheme.codes, dir_scheme.offsets):
        offset_lookup[code, :] = [di, dj]

    dirs = np.array([[3, 3, 3], [3, 3, 3], [1, 1, 0]], dtype=np.uint8)
    x, y = np.meshgrid(
        np.arange(dirs.shape[1], dtype=np.float32),
        np.arange(dirs.shape[0], dtype=np.float32),
        indexing="xy",
    )

    common_kwargs = {
        "dirs": dirs.astype(np.uint8, order="F"),
        "x": x.astype(np.float32, order="F"),
        "y": y.astype(np.float32, order="F"),
        "offset_lookup": offset_lookup,
        "check_flag": True,
    }

    dists, err_code = ridges_f.compute_confluence_dist([1, 1], [1, 2], **common_kwargs)
    assert np.isclose(dists[0], 3.0)
    assert np.isclose(dists[1], 2.0)
    dists, err_code = ridges_f.compute_confluence_dist([1, 1], [1, 3], **common_kwargs)
    assert np.isclose(dists[0], 4.0)
    assert np.isclose(dists[1], 2.0)
    dists, err_code = ridges_f.compute_confluence_dist([3, 1], [3, 3], **common_kwargs)
    assert np.isclose(dists[0], 2.0)
    assert np.isclose(dists[1], 0.0)

    common_kwargs["dirs"] = np.array(
        [[5, 1, 1], [5, 1, 1], [5, 1, 1]], dtype=np.uint8, order="F"
    )
    dists, err_code = ridges_f.compute_confluence_dist([1, 1], [1, 2], **common_kwargs)
    assert np.isclose(dists[0], 0.0)
    assert np.isclose(dists[1], 1.0)


def test_confluence_distance_reports_cyclic_path():
    dir_scheme = D8Directions(transform_codes=lambda x: x)
    offset_lookup = np.zeros((256, 2), dtype=np.int32)
    codes_by_offset = {}
    for code, offset in zip(dir_scheme.codes, dir_scheme.offsets):
        offset_lookup[code, :] = offset
        codes_by_offset[tuple(offset)] = code

    dirs = np.array(
        [
            [codes_by_offset[(0, 1)], codes_by_offset[(0, -1)], 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
        order="F",
    )
    x, y = np.meshgrid(
        np.arange(dirs.shape[1], dtype=np.float32),
        np.arange(dirs.shape[0], dtype=np.float32),
        indexing="xy",
    )

    _, err_code = ridges_f.compute_confluence_dist(
        [1, 1],
        [2, 3],
        dirs,
        x.astype(np.float32, order="F"),
        y.astype(np.float32, order="F"),
        offset_lookup,
        True,
    )

    assert err_code == 1


def test_confluence_distance_defaults_to_checking_for_confluence():
    dirs = np.array([[128, 0], [128, 0]], dtype=np.uint8, order="F")
    x, y = np.meshgrid(
        np.arange(dirs.shape[1], dtype=np.float32),
        np.arange(dirs.shape[0], dtype=np.float32),
        indexing="xy",
    )
    offset_lookup = np.zeros((256, 2), dtype=np.int32, order="F")
    offset_lookup[128] = [0, 1]

    default_dists, default_err = ridges_f.compute_confluence_dist(
        [1, 1], [2, 1], dirs, x, y, offset_lookup
    )
    explicit_dists, explicit_err = ridges_f.compute_confluence_dist(
        [1, 1], [2, 1], dirs, x, y, offset_lookup, True
    )

    assert default_err == explicit_err == 0
    np.testing.assert_allclose(default_dists, explicit_dists)
    np.testing.assert_allclose(default_dists, [1.0, 1.0])


def test_confluence_distance_accepts_unsigned_direction_codes():
    dirs = np.array([[255, 0], [255, 0]], dtype=np.uint8, order="F")
    x, y = np.meshgrid(
        np.arange(dirs.shape[1], dtype=np.float32),
        np.arange(dirs.shape[0], dtype=np.float32),
        indexing="xy",
    )
    offset_lookup = np.zeros((256, 2), dtype=np.int32, order="F")
    offset_lookup[255] = [0, 1]

    dists, err_code = ridges_f.compute_confluence_dist(
        [1, 1], [2, 1], dirs, x, y, offset_lookup
    )

    assert err_code == 0
    np.testing.assert_allclose(dists, [1.0, 1.0])


def test_max_branch_distance_translates_allocation_failure(monkeypatch):
    def fake_compute(*args):
        return np.zeros((1, 1), dtype=np.float32), 2

    monkeypatch.setattr(
        ridges_m, "ridges_f", SimpleNamespace(compute_max_branch_dist=fake_compute)
    )

    with pytest.raises(MemoryError, match=r"compute_max_branch_dist.*error code 2"):
        ridges_m.compute_dist2conf_max(np.zeros((1, 1), dtype=np.uint8))
