"""
Tests digital elevation model preprocessing using the FORTRAN
backend.

Last modified: 2026-08-22, En-Chi Lee (williameclee@gmail.com)
"""

from tests.core import *

import pytest
import numpy as np

from formosa import D8Directions
import formosa.geomorphology.drainage.preprocessing as preproc_m
from formosa.geomorphology._native import drainage_preprocessing as preproc_f

from types import SimpleNamespace


def test_detect_ocean_basins_labels_separate_boundary_components():
    dem = np.full((7, 9), 5.0, dtype=np.float32)
    dem[0:3, 0:2] = 0.0
    dem[4:7, 7:9] = -1.0
    dem[3, 4] = 0.0  # Enclosed low cell is not an ocean basin.

    basins = preproc_m.detect_ocean_basins_from_boundary(dem)

    left_label = basins[0, 0]
    right_label = basins[-1, -1]
    assert left_label > 0
    assert right_label > 0
    assert left_label != right_label
    assert np.all(basins[0:3, 0:2] == left_label)
    assert np.all(basins[4:7, 7:9] == right_label)
    assert basins[3, 4] == 0
    assert np.all(basins[dem == 5.0] == 0)


def test_detect_ocean_basins_respects_valids_and_exact_level_mode():
    dem = np.array(
        [[0.0, -1.0, -1.0, 5.0], [0.0, -1.0, -1.0, 5.0]],
        dtype=np.float32,
    )
    valids = np.ones(dem.shape, dtype=bool)
    valids[1, 0] = False

    exact = preproc_m.detect_ocean_basins_from_boundary(
        dem, valids=valids, flood_below=False
    )
    flooded = preproc_m.detect_ocean_basins_from_boundary(
        dem, valids=valids, flood_below=True
    )

    assert exact[0, 0] > 0
    assert not np.any(exact[:, 1:3])
    assert exact[1, 0] == 0
    assert np.all(flooded[:, 1:3] > 0)
    assert flooded[1, 0] == 0


def test_invalidate_ocean_basins_filters_by_inclusive_size():
    dem = np.full((7, 9), 5.0, dtype=np.float32)
    dem[0:2, 0:3] = 0.0  # Six-cell boundary basin.
    dem[5:7, 7:9] = 0.0  # Four-cell boundary basin.
    dem[3, 3:6] = 0.0  # Enclosed basin is never an ocean candidate.
    valids = np.ones(dem.shape, dtype=bool)
    valids[6, 0] = False

    result = preproc_m.invalidate_ocean_basins(dem, valids=valids, min_size=6)

    assert not np.any(result[0:2, 0:3])
    assert np.all(result[5:7, 7:9])
    assert np.all(result[3, 3:6])
    assert not result[6, 0]
    np.testing.assert_array_equal(valids[0:2, 0:3], np.ones((2, 3), dtype=bool))


@pytest.mark.parametrize("minimum_basin_size", [0, -1])
def test_invalidate_ocean_basins_rejects_nonpositive_size(minimum_basin_size):
    with pytest.raises(ValueError, match="at least 1"):
        preproc_m.invalidate_ocean_basins(
            np.zeros((2, 2), dtype=np.float32),
            min_size=minimum_basin_size,
        )


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (6, 1), (2, 8), (9, 2)])
def test_detect_ocean_basins_handles_boundary_only_grids(shape):
    ocean = np.zeros(shape, dtype=np.float32)
    land = np.ones(shape, dtype=np.float32)

    ocean_basins = preproc_m.detect_ocean_basins_from_boundary(ocean)
    land_basins = preproc_m.detect_ocean_basins_from_boundary(land)

    assert np.all(ocean_basins == ocean_basins.flat[0])
    assert ocean_basins.flat[0] > 0
    assert not np.any(land_basins)


def test_detect_ocean_basins_respects_connectivity_scheme():
    dem = np.full((3, 3), 5.0, dtype=np.float32)
    dem[0, 0] = 0.0
    dem[1, 1] = 0.0
    d4 = SimpleNamespace(
        offsets=np.array([[-1, 0], [0, -1], [0, 1], [1, 0]], dtype=np.int32)
    )

    d8_basins = preproc_m.detect_ocean_basins_from_boundary(dem)
    d4_basins = preproc_m.detect_ocean_basins_from_boundary(dem, dir_scheme=d4)  # type: ignore

    assert d8_basins[1, 1] == d8_basins[0, 0]
    assert d4_basins[0, 0] > 0
    assert d4_basins[1, 1] == 0


def test_detect_ocean_basins_excludes_nonfinite_and_all_invalid_cells():
    dem = np.array([[0.0, np.nan], [np.inf, -np.inf]], dtype=np.float32)
    basins = preproc_m.detect_ocean_basins_from_boundary(dem)
    all_invalid = preproc_m.detect_ocean_basins_from_boundary(
        np.zeros((2, 3), dtype=np.float32),
        valids=np.zeros((2, 3), dtype=bool),
    )

    assert basins[0, 0] > 0
    assert not np.any(basins[~np.isfinite(dem)])
    assert not np.any(all_invalid)


@pytest.mark.parametrize("dtype", [np.int16, np.float32, np.float64])
def test_detect_ocean_basins_accepts_numeric_dtypes_and_noncontiguous_views(dtype):
    source = np.zeros((8, 10), dtype=dtype)
    dem = source[::2, ::2]
    original = dem.copy()

    basins = preproc_m.detect_ocean_basins_from_boundary(dem)

    assert basins.shape == dem.shape
    assert basins.dtype == np.int32
    assert np.all(basins > 0)
    np.testing.assert_array_equal(dem, original)


@pytest.mark.parametrize("min_size", [True, 1.5, "2"])
def test_invalidate_ocean_basins_rejects_noninteger_size(min_size):
    with pytest.raises(TypeError, match="integer"):
        preproc_m.invalidate_ocean_basins(
            np.zeros((2, 2), dtype=np.float32), min_size=min_size
        )


@pytest.mark.parametrize("ocean_level", [np.nan, np.inf, -np.inf])
def test_detect_ocean_basins_rejects_nonfinite_ocean_level(ocean_level):
    with pytest.raises(ValueError, match="finite"):
        preproc_m.detect_ocean_basins_from_boundary(
            np.zeros((2, 2), dtype=np.float32), ocean_level=ocean_level
        )


def test_detect_ocean_basins_rejects_nonboolean_flood_below_and_complex_dem():
    dem = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(TypeError, match="boolean"):
        preproc_m.detect_ocean_basins_from_boundary(dem, flood_below="false")  # type: ignore
    with pytest.raises(TypeError, match="real-valued"):
        preproc_m.detect_ocean_basins_from_boundary(dem.astype(np.complex64))  # type: ignore


def test_fill_depressions():
    dem = np.array(
        [[5.0, 5.0, 5.0], [5.0, 1.0, 5.0], [5.0, 5.0, 5.0]],
        dtype=np.float64,
    )

    filled = preproc_m.fill_depressions(dem)

    np.testing.assert_array_equal(filled, np.full((3, 3), 5.0))
    assert filled.dtype == dem.dtype
    assert dem[1, 1] == 1.0


@pytest.mark.parametrize(
    ("dem", "expected"),
    [
        (
            [
                [9, 9, 9, 9, 9],
                [9, 3, 3, 3, 9],
                [9, 3, 1, 3, 4],
                [9, 3, 3, 3, 9],
                [9, 9, 9, 9, 9],
            ],
            [
                [9, 9, 9, 9, 9],
                [9, 4, 4, 4, 9],
                [9, 4, 4, 4, 4],
                [9, 4, 4, 4, 9],
                [9, 9, 9, 9, 9],
            ],
        ),
        (
            [[0, 0, 0, 0], [4, -3, -2, 0], [4, -4, -1, 0], [4, 4, 4, 4]],
            [[0, 0, 0, 0], [4, 0, 0, 0], [4, 0, 0, 0], [4, 4, 4, 4]],
        ),
        (
            [[5, 5, 5, 5], [5, 2, 2, 1], [5, 2, 0, 5], [5, 5, 5, 5]],
            [[5, 5, 5, 5], [5, 2, 2, 1], [5, 2, 1, 5], [5, 5, 5, 5]],
        ),
    ],
    ids=["single-spill-basin", "negative-basin", "open-channel"],
)
def test_fill_depressions_fortran_reference_terrains(dem, expected):
    filled = preproc_m.fill_depressions(np.asarray(dem, dtype=np.float32))

    np.testing.assert_array_equal(filled, np.asarray(expected, dtype=np.float32))


@pytest.mark.parametrize("dtype", [np.int16, np.float32, np.float64])
def test_fill_depressions_preserves_shape_dtype_and_input(dtype):
    source = np.array(
        [[7, 7, 7, 7], [7, 1, 2, 7], [7, 3, 0, 7], [7, 7, 7, 7]],
        dtype=dtype,
    )
    dem = source[:, ::-1]
    original = dem.copy()

    filled = preproc_m.fill_depressions(dem)

    assert filled.shape == dem.shape
    assert filled.dtype == dem.dtype
    np.testing.assert_array_equal(dem, original)
    np.testing.assert_array_equal(filled, np.full(dem.shape, 7, dtype=dtype))


@pytest.mark.parametrize("shape", [(1, 7), (6, 1), (2, 8), (9, 2)])
def test_fill_depressions_boundary_only_grids_are_unchanged(shape):
    dem = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

    filled = preproc_m.fill_depressions(dem)

    np.testing.assert_array_equal(filled, dem)


def test_fill_depressions_is_monotonic_and_idempotent_randomly():
    rng = np.random.default_rng(8675309)
    for _ in range(100):
        shape = tuple(rng.integers(3, 20, size=2))
        dem = rng.integers(-1000, 1000, size=shape).astype(np.float32)

        filled = preproc_m.fill_depressions(dem)
        filled_twice = preproc_m.fill_depressions(filled)

        assert np.all(filled >= dem)
        np.testing.assert_array_equal(filled[0, :], dem[0, :])
        np.testing.assert_array_equal(filled[-1, :], dem[-1, :])
        np.testing.assert_array_equal(filled[:, 0], dem[:, 0])
        np.testing.assert_array_equal(filled[:, -1], dem[:, -1])
        np.testing.assert_array_equal(filled_twice, filled)


def test_fill_depressions_validates_mask_shape():
    with pytest.raises(ValueError, match="Shapes .* must match"):
        preproc_m.fill_depressions(
            np.ones((3, 3), dtype=np.float32),
            valids=np.ones((2, 2), dtype=bool),
        )


def test_fill_depressions_fortran_preserves_invalid_cells():
    dem = np.array(
        [[5.0, 5.0, 5.0], [5.0, -9999.0, 1.0], [5.0, 5.0, 5.0]],
        dtype=np.float32,
    )
    valids = np.ones(dem.shape, dtype=bool)
    valids[1, 1] = False

    filled = preproc_m.fill_depressions(dem, valids=valids)

    assert filled[1, 1] == dem[1, 1]


def test_fill_depressions_treats_internal_invalids_as_outlets():
    dem = np.full((5, 5), 5.0, dtype=np.float32)
    dem[1:4, 1:4] = 1.0
    dem[2, 2] = -9999.0
    valids = np.ones(dem.shape, dtype=bool)
    valids[2, 2] = False

    filled = preproc_m.fill_depressions(dem, valids=valids)

    np.testing.assert_array_equal(filled, dem)


def test_fill_depressions_treats_boundary_invalids_as_outlets():
    dem = np.full((5, 5), -9999.0, dtype=np.float32)
    dem[1:4, 1:4] = 5.0
    dem[2, 2] = 1.0
    valids = np.zeros(dem.shape, dtype=bool)
    valids[1:4, 1:4] = True

    filled = preproc_m.fill_depressions(dem, valids=valids)

    np.testing.assert_array_equal(filled[valids], np.full(valids.sum(), 5.0))
    np.testing.assert_array_equal(filled[~valids], dem[~valids])


def test_fill_depressions_uses_diagonal_invalid_adjacency():
    dem = np.full((5, 5), 10.0, dtype=np.float32)
    dem[1, 1] = 1.0
    dem[2, 2] = -9999.0
    valids = np.ones(dem.shape, dtype=bool)
    valids[2, 2] = False

    filled = preproc_m.fill_depressions(dem, valids=valids)

    assert filled[1, 1] == 1.0
    assert filled[2, 2] == -9999.0


def test_fill_depressions_deduplicates_outlets_adjacent_to_many_invalids():
    rows, cols = np.indices((9, 9))
    valids = (rows + cols) % 2 == 0
    dem = np.arange(81, dtype=np.float32).reshape(9, 9)
    dem[~valids] = -9999.0

    filled = preproc_m.fill_depressions(dem, valids=valids)

    np.testing.assert_array_equal(filled, dem)


def test_fill_depressions_fills_valid_island_surrounded_by_invalids():
    dem = np.full((9, 9), -9999.0, dtype=np.float32)
    dem[2:7, 2:7] = 3.0
    dem[3:6, 3:6] = 1.0
    valids = np.zeros(dem.shape, dtype=bool)
    valids[2:7, 2:7] = True

    filled = preproc_m.fill_depressions(dem, valids=valids)

    np.testing.assert_array_equal(filled[valids], np.full(valids.sum(), 3.0))
    np.testing.assert_array_equal(filled[~valids], dem[~valids])


def test_fill_depressions_all_invalid_is_unchanged():
    dem = np.arange(12, dtype=np.float32).reshape(3, 4)

    filled = preproc_m.fill_depressions(dem, valids=np.zeros(dem.shape, dtype=bool))

    np.testing.assert_array_equal(filled, dem)
    assert filled is not dem


def test_fill_depressions_processes_multiple_large_basins_together():
    dem = np.full((7, 13), 10.0, dtype=np.float32)
    dem[1:6, 1:6] = 5.0
    dem[1:6, 7:12] = 5.0
    dem[3, 3] = 0.0
    dem[3, 9] = -1.0
    dem[1, 1] = 1.0
    dem[1, 11] = 2.0

    filled = preproc_m.fill_depressions(dem, max_fill_size=24)

    expected = dem.copy()
    expected[1, 1] = 5.0
    expected[1, 11] = 5.0
    np.testing.assert_array_equal(filled, expected)


def test_label_mask_areas():
    dir_scheme = D8Directions()
    mask = np.array(
        [
            [T, F, T, F, T],
            [T, F, T, F, T],
            [T, F, T, F, T],
            [T, F, T, F, T],
        ],
        dtype=bool,
    )
    labels, err_code = preproc_f.label_mask_areas(mask, dir_scheme.offsets)
    assert err_code == 0
    assert ~np.any(labels[:, 1])
    assert ~np.any(labels[:, 3])
    c1 = np.unique(labels[:, 0])
    assert np.size(c1) == 1
    c2 = np.unique(labels[:, 2])
    assert np.size(c2) == 1
    c3 = np.unique(labels[:, 4])
    assert np.size(c3) == 1
    assert c1 != c2
    assert c1 != c3
    assert c2 != c3

    mask = np.array(
        [
            [T, F, T, F, T],
            [T, F, T, F, T],
            [T, T, T, F, T],
            [T, F, T, F, T],
        ],
        dtype=bool,
    )
    labels, err_code = preproc_f.label_mask_areas(mask, dir_scheme.offsets)
    assert err_code == 0
    assert ~np.any(labels[:, 3])
    c1 = np.unique(labels[:, 0])
    assert np.size(c1) == 1
    c2 = np.unique(labels[:, 2])
    assert np.size(c2) == 1
    c3 = np.unique(labels[:, 4])
    assert np.size(c3) == 1
    assert c1 == c2
    assert c1 != c3
