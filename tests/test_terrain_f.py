"""
Tests terrain metrics using the Fortran backend.

This module compares public isolation and prominence results with
exhaustive reference calculations and covers input validation.

Created: 2026-08-19, En-Chi Lee (williameclee@gmail.com)
Last modified: 2026-08-22, En-Chi Lee (williameclee@gmail.com)
"""

import pytest

import heapq
import numpy as np

from formosa.geomorphology.drainage.directions import D8Directions
from formosa.geomorphology.terrain import compute_isolation, compute_prominence


def _compute_prominence_labels(*args, **kwargs):
    """
    Returns prominence outputs with separate peak and saddle rasters.
    """
    proms, feats, feat_types, _, key_saddles, feat_prnts = compute_prominence(
        *args, **kwargs
    )
    has_feat = feats >= 0
    feat_labels = np.where(has_feat, feats + 1, 0)
    peak_cells = np.zeros(feats.shape, dtype=bool)
    saddle_cells = np.zeros(feats.shape, dtype=bool)
    peak_cells[has_feat] = feat_types[feats[has_feat]] == 1
    saddle_cells[has_feat] = feat_types[feats[has_feat]] == 2
    peaks = np.where(peak_cells, feat_labels, 0)
    saddles = np.where(saddle_cells, feat_labels, 0)
    return proms, peaks, saddles, key_saddles, feat_prnts


def _brute_force_isolation(
    dem: np.ndarray, valids: np.ndarray, dx: float, dy: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns exact isolation distances, ILP existence, and censoring.
    """
    nrows, ncols = dem.shape
    rows, cols = np.indices(dem.shape)
    isos = np.zeros(dem.shape, dtype=np.float32)
    has_ilp = np.zeros(dem.shape, dtype=bool)

    for ci in range(nrows):
        for cj in range(ncols):
            if not valids[ci, cj]:
                continue

            higher = valids & (dem > dem[ci, cj])
            if not np.any(higher):
                continue

            dist2 = ((rows[higher] - ci) * dy) ** 2 + ((cols[higher] - cj) * dx) ** 2  # type: ignore
            isos[ci, cj] = np.sqrt(np.min(dist2))
            has_ilp[ci, cj] = True

    row_margin = (np.minimum(rows, nrows - 1 - rows).astype(np.float64) + 0.5) * dy
    col_margin = (np.minimum(cols, ncols - 1 - cols).astype(np.float64) + 0.5) * dx
    boundary_distance = np.minimum(row_margin, col_margin)
    censored = valids & (~has_ilp | (isos > boundary_distance))

    return isos, has_ilp, censored


def _brute_force_prominence(
    dem: np.ndarray,
    valids: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    """Returns prominence from exhaustive maximum-bottleneck paths."""
    nrows, ncols = dem.shape
    proms = np.full(dem.shape, -1.0, dtype=np.float32)
    proms[valids] = 0.0
    visited: set[tuple[int, int]] = set()

    for start_i in range(nrows):
        for start_j in range(ncols):
            if not valids[start_i, start_j] or (start_i, start_j) in visited:
                continue

            zpeak = dem[start_i, start_j]
            plateau = {(start_i, start_j)}
            queue = [(start_i, start_j)]
            while queue:
                ci, cj = queue.pop()
                for di, dj in offsets:
                    ni = ci + int(di)
                    nj = cj + int(dj)
                    neighbour = (ni, nj)
                    if not (0 <= ni < nrows and 0 <= nj < ncols):
                        continue
                    if not valids[ni, nj] or dem[ni, nj] != zpeak:
                        continue
                    if neighbour in plateau:
                        continue
                    plateau.add(neighbour)
                    queue.append(neighbour)
            visited.update(plateau)

            has_higher_neighbour = any(
                0 <= ci + int(di) < nrows
                and 0 <= cj + int(dj) < ncols
                and valids[ci + int(di), cj + int(dj)]
                and dem[ci + int(di), cj + int(dj)] > zpeak
                for ci, cj in plateau
                for di, dj in offsets
            )
            if has_higher_neighbour:
                continue

            for ci, cj in plateau:
                proms[ci, cj] = -1.0

            capacities = np.full(dem.shape, -np.inf, dtype=np.float64)
            priority_queue: list[tuple[float, int, int]] = []
            for ci, cj in plateau:
                capacities[ci, cj] = zpeak
                heapq.heappush(priority_queue, (-float(zpeak), ci, cj))

            saddle = None
            while priority_queue:
                negative_capacity, ci, cj = heapq.heappop(priority_queue)
                capacity = -negative_capacity
                if capacity != capacities[ci, cj]:
                    continue
                if dem[ci, cj] > zpeak:
                    saddle = capacity
                    break

                for di, dj in offsets:
                    ni = ci + int(di)
                    nj = cj + int(dj)
                    if not (0 <= ni < nrows and 0 <= nj < ncols):
                        continue
                    if not valids[ni, nj]:
                        continue

                    candidate = min(capacity, float(dem[ni, nj]))
                    if candidate <= capacities[ni, nj]:
                        continue
                    capacities[ni, nj] = candidate
                    heapq.heappush(priority_queue, (-candidate, ni, nj))

            if saddle is not None:
                for ci, cj in plateau:
                    proms[ci, cj] = zpeak - saddle

    return proms


def _assert_label_raster(labels: np.ndarray, offsets: np.ndarray) -> None:
    """Checks that positive IDs are consecutive connected regions."""
    positive_ids = np.unique(labels[labels > 0])
    np.testing.assert_array_equal(
        positive_ids,
        np.arange(1, len(positive_ids) + 1, dtype=positive_ids.dtype),
    )

    nrows, ncols = labels.shape
    for feature_id in positive_ids:
        feature_cells = {tuple(cell) for cell in np.argwhere(labels == feature_id)}
        reached = {next(iter(feature_cells))}
        queue = list(reached)
        while queue:
            ci, cj = queue.pop()
            for di, dj in offsets:
                neighbour = (ci + int(di), cj + int(dj))
                ni, nj = neighbour
                if not (0 <= ni < nrows and 0 <= nj < ncols):
                    continue
                if neighbour not in feature_cells or neighbour in reached:
                    continue
                reached.add(neighbour)
                queue.append(neighbour)

        assert reached == feature_cells


@pytest.mark.parametrize(
    ("shape", "dx", "dy", "include_invalids", "seed"),
    [
        pytest.param((5, 5), 1.0, 1.0, False, 1, id="square-isotropic-valid"),
        pytest.param((6, 6), 4.0, 0.5, True, 2, id="square-anisotropic"),
        pytest.param((4, 7), 1.0, 1.0, True, 3, id="wide-isotropic"),
        pytest.param((4, 7), 8.0, 0.75, False, 4, id="wide-anisotropic"),
        pytest.param((7, 4), 0.5, 6.0, True, 5, id="tall-anisotropic"),
        pytest.param((1, 9), 2.5, 1.0, True, 6, id="single-row"),
        pytest.param((9, 1), 1.0, 2.5, True, 7, id="single-column"),
    ],
)
def test_calculate_isolation_matches_brute_force(shape, dx, dy, include_invalids, seed):
    rng = np.random.default_rng(seed)
    dem = rng.integers(-5, 20, size=shape).astype(np.float32)
    valids = np.ones(shape, dtype=bool)
    if include_invalids:
        valids.flat[::4] = False
        valids.flat[-1] = True

    expected_isos, expected_has_ilp, expected_censored = _brute_force_isolation(
        dem, valids, dx, dy
    )
    isos, ilpis, ilpjs, censored = compute_isolation(dem, valids, dx=dx, dy=dy)

    np.testing.assert_allclose(isos, expected_isos, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(censored, expected_censored)
    assert np.all(ilpis[~expected_has_ilp] == -1)
    assert np.all(ilpjs[~expected_has_ilp] == -1)

    for ci, cj in np.argwhere(expected_has_ilp):
        ilpi = ilpis[ci, cj]
        ilpj = ilpjs[ci, cj]
        assert 0 <= ilpi < shape[0]
        assert 0 <= ilpj < shape[1]
        assert valids[ilpi, ilpj]
        assert dem[ilpi, ilpj] > dem[ci, cj]
        ilp_distance = np.hypot((ilpi - ci) * dy, (ilpj - cj) * dx)
        assert ilp_distance == pytest.approx(expected_isos[ci, cj], rel=1e-6)


def test_anisotropic_spacing_can_make_distant_row_cell_nearest():
    dem = np.zeros((6, 6), dtype=np.float32)
    dem[2, 2] = 5.0
    dem[2, 3] = 6.0  # One column away: distance 10.
    dem[4, 2] = 7.0  # Two rows away: distance 2.

    isos, ilpis, ilpjs, censored = compute_isolation(dem, dx=10.0, dy=1.0)

    assert isos[2, 2] == pytest.approx(2.0)
    assert (ilpis[2, 2], ilpjs[2, 2]) == (4, 2)
    assert not censored[2, 2]


def test_equal_elevations_do_not_qualify_as_higher():
    dem = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 5.0, 5.0, 1.0],
            [1.0, 5.0, 5.0, 6.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )

    isos, ilpis, ilpjs, censored = compute_isolation(dem)

    assert isos[1, 1] == pytest.approx(np.sqrt(5.0))
    assert (ilpis[1, 1], ilpjs[1, 1]) == (2, 3)
    assert isos[2, 2] == pytest.approx(1.0)
    assert (ilpis[2, 2], ilpjs[2, 2]) == (2, 3)
    assert isos[2, 3] == 0.0
    assert (ilpis[2, 3], ilpjs[2, 3]) == (-1, -1)
    assert censored[2, 3]


def test_nonfinite_and_masked_higher_cells_are_ignored():
    dem = np.array(
        [
            [9.0, 0.0, 7.0],
            [0.0, 5.0, np.nan],
            [0.0, 0.0, 6.0],
        ],
        dtype=np.float32,
    )
    valids = np.ones(dem.shape, dtype=bool)
    valids[0, 0] = False

    isos, ilpis, ilpjs, censored = compute_isolation(dem, valids)

    assert isos[1, 1] == pytest.approx(np.sqrt(2.0))
    assert (ilpis[1, 1], ilpjs[1, 1]) == (2, 2)
    assert isos[0, 0] == 0.0
    assert (ilpis[0, 0], ilpjs[0, 0]) == (-1, -1)
    assert not censored[0, 0]
    assert isos[1, 2] == 0.0
    assert (ilpis[1, 2], ilpjs[1, 2]) == (-1, -1)
    assert not censored[1, 2]


@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (5, 4)])
def test_all_invalid_cells_have_no_isolation_limit_point(shape):
    dem = np.ones(shape, dtype=np.float32)
    valids = np.zeros(shape, dtype=bool)

    isos, ilpis, ilpjs, censored = compute_isolation(dem, valids)

    assert np.all(isos == 0.0)
    assert np.all(ilpis == -1)
    assert np.all(ilpjs == -1)
    assert not np.any(censored)


def test_censoring_uses_outer_raster_footprint():
    dem = np.zeros((5, 5), dtype=np.float32)
    dem[2, 2] = 5.0
    dem[0, 0] = 6.0

    isos, _, _, censored = compute_isolation(dem)

    # The centre is 2.5 cells from the raster footprint, but its ILP
    # is sqrt(8) cells away at the corner.
    assert isos[2, 2] == pytest.approx(np.sqrt(8.0))
    assert censored[2, 2]

    # Every positive-radius search from a boundary-cell centre extends
    # beyond its half-cell-wide footprint margin.
    assert isos[0, 1] == pytest.approx(1.0)
    assert censored[0, 1]


def test_isolation_circle_inside_raster_footprint_is_not_censored():
    dem = np.zeros((7, 7), dtype=np.float32)
    dem[3, 3] = 5.0
    dem[3, 5] = 6.0

    isos, _, _, censored = compute_isolation(dem)

    assert isos[3, 3] == pytest.approx(2.0)
    assert not censored[3, 3]


@pytest.mark.parametrize(
    ("name", "value", "exception"),
    [
        pytest.param("dx", 0.0, ValueError, id="zero-dx"),
        pytest.param("dy", -1.0, ValueError, id="negative-dy"),
        pytest.param("dx", np.nan, ValueError, id="nan-dx"),
        pytest.param("dy", np.inf, ValueError, id="infinite-dy"),
        pytest.param("dx", None, TypeError, id="none-dx"),
        pytest.param("dy", True, TypeError, id="boolean-dy"),
        pytest.param("dx", "1", TypeError, id="string-dx"),
        pytest.param("dy", 1 + 0j, TypeError, id="complex-dy"),
        pytest.param("dx", [1.0], TypeError, id="array-dx"),
    ],
)
def test_calculate_isolation_rejects_invalid_spacing(name, value, exception):
    kwargs = {name: value}
    with pytest.raises(exception):
        compute_isolation(np.ones((2, 2), dtype=np.float32), **kwargs)


@pytest.mark.parametrize(
    ("dem", "expected"),
    [
        pytest.param(
            [[3, 1, 2]],
            [[-1, 0, 1]],
            id="unequal-peaks",
        ),
        pytest.param(
            [[2, 1, 2]],
            [[-1, 0, -1]],
            id="global-copeaks",
        ),
        pytest.param(
            [[4, 2, 4, 2, 5]],
            [[2, 0, 2, 0, -1]],
            id="equal-lower-peaks",
        ),
        pytest.param(
            [
                [0, 0, 0, 0, 0],
                [0, 4, 4, 1, 5],
                [0, 4, 4, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 3, 3, 0, -1],
                [0, 3, 3, 0, 0],
            ],
            id="summit-plateau",
        ),
    ],
)
def test_compute_prominence_known_landforms(dem, expected):
    proms, peaks, saddles, _, _ = _compute_prominence_labels(
        np.asarray(dem, dtype=np.float32)
    )

    np.testing.assert_array_equal(proms, expected)
    assert np.all(peaks[np.asarray(expected) != 0] > 0)
    assert np.all(peaks[np.asarray(expected) == 0] == 0)
    assert not np.any((peaks > 0) & (saddles > 0))
    _assert_label_raster(peaks + saddles, D8Directions().offsets)


def test_compute_prominence_labels_peak_and_key_saddle_plateaus():
    dem = np.array([[5.0, 1.0, 1.0, 4.0]], dtype=np.float32)

    proms, peaks, saddles, _, _ = _compute_prominence_labels(dem)

    np.testing.assert_array_equal(proms, [[-1.0, 0.0, 0.0, 3.0]])
    assert peaks[0, 0] > 0
    assert peaks[0, 3] > 0
    assert peaks[0, 0] != peaks[0, 3]
    assert np.all(peaks[0, 1:3] == 0)
    assert saddles[0, 1] > 0
    assert saddles[0, 1] == saddles[0, 2]
    assert saddles[0, 0] == saddles[0, 3] == 0


def test_compute_prominence_labels_multiway_key_saddle_once():
    dem = np.array(
        [
            [0.0, 5.0, 0.0],
            [4.0, 1.0, 3.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    dir_scheme = D8Directions()
    dir_scheme.offsets = np.array(
        [[-1, 0], [0, -1], [0, 0], [0, 1], [1, 0]], dtype=np.int32, order="F"
    )

    _, peaks, saddles, _, _ = _compute_prominence_labels(dem, dir_scheme=dir_scheme)

    assert len(np.unique(peaks[peaks > 0])) == 3
    assert saddles[1, 1] > 0
    assert np.count_nonzero(saddles) == 1


def test_peak_and_saddle_ids_use_universal_namespace():
    dem = np.array([[3.0, 1.0, 2.0]], dtype=np.float32)

    _, peaks, saddles, _, _ = _compute_prominence_labels(dem)

    assert 1 in peaks
    assert set(peaks[peaks > 0]).isdisjoint(saddles[saddles > 0])


@pytest.mark.parametrize(
    ("shape", "include_invalids", "include_flats", "seed"),
    [
        pytest.param((5, 5), False, False, 11, id="square-distinct"),
        pytest.param((5, 5), False, True, 12, id="square-flats"),
        pytest.param((4, 7), True, False, 13, id="wide-masked"),
        pytest.param((7, 4), True, True, 14, id="tall-masked-flats"),
        pytest.param((1, 9), True, True, 15, id="single-row"),
        pytest.param((9, 1), True, False, 16, id="single-column"),
    ],
)
def test_compute_prominence_matches_brute_force(
    shape, include_invalids, include_flats, seed
):
    rng = np.random.default_rng(seed)
    if include_flats:
        dem = rng.integers(-5, 8, size=shape).astype(np.float32)
    else:
        dem = rng.permutation(np.prod(shape)).reshape(shape).astype(np.float32)

    valids = np.ones(shape, dtype=bool)
    if include_invalids:
        valids.flat[::4] = False
        valids.flat[-1] = True

    dir_scheme = D8Directions()
    expected = _brute_force_prominence(dem, valids, dir_scheme.offsets)
    proms, peaks, saddles, _, _ = _compute_prominence_labels(dem, valids, dir_scheme)

    np.testing.assert_array_equal(proms, expected)
    assert np.all(peaks[~valids] == 0)
    assert np.all(saddles[~valids] == 0)
    assert not np.any((peaks > 0) & (saddles > 0))
    for feature_id in np.unique(peaks[peaks > 0]):
        assert np.unique(dem[peaks == feature_id]).size == 1
    for feature_id in np.unique(saddles[saddles > 0]):
        assert np.unique(dem[saddles == feature_id]).size == 1
    _assert_label_raster(peaks + saddles, dir_scheme.offsets)


def test_compute_prominence_marks_disconnected_component_maxima():
    dem = np.array([[5.0, 0.0, 4.0]], dtype=np.float32)
    valids = np.array([[True, False, True]])

    proms, peaks, saddles, _, _ = _compute_prominence_labels(dem, valids)

    np.testing.assert_array_equal(proms, [[-1.0, -1.0, -1.0]])
    assert peaks[0, 0] > 0
    assert peaks[0, 2] > 0
    assert peaks[0, 0] != peaks[0, 2]
    assert peaks[0, 1] == 0
    assert not np.any(saddles)


def test_compute_prominence_treats_nonfinite_cells_as_invalid():
    dem = np.array([[5.0, np.nan, 4.0, np.inf]], dtype=np.float32)

    proms, peaks, saddles, _, _ = _compute_prominence_labels(dem)

    np.testing.assert_array_equal(proms, [[-1.0, -1.0, -1.0, -1.0]])
    assert peaks[0, 0] > 0
    assert peaks[0, 2] > 0
    assert peaks[0, 0] != peaks[0, 2]
    assert np.all(peaks[0, [1, 3]] == 0)
    assert not np.any(saddles)


@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (5, 4)])
def test_compute_prominence_marks_all_invalid_cells(shape):
    dem = np.ones(shape, dtype=np.float32)
    valids = np.zeros(shape, dtype=bool)

    proms, peaks, saddles, saddle_lookup, feat_tree = _compute_prominence_labels(
        dem, valids
    )

    assert np.all(proms == -1.0)
    assert not np.any(peaks)
    assert not np.any(saddles)
    assert saddle_lookup.size == 0
    assert feat_tree.size == 0


def test_compute_prominence_respects_direction_connectivity():
    dem = np.array([[5.0, 0.0], [0.0, 4.0]], dtype=np.float32)
    dir_scheme = D8Directions()
    dir_scheme.offsets = np.array(
        [[-1, 0], [0, -1], [0, 0], [0, 1], [1, 0]],
        dtype=np.int32,
        order="F",
    )

    d8_proms, d8_peaks, d8_saddles, _, _ = _compute_prominence_labels(dem)
    cardinal_proms, cardinal_peaks, cardinal_saddles, _, _ = _compute_prominence_labels(
        dem, dir_scheme=dir_scheme
    )

    np.testing.assert_array_equal(d8_proms, [[-1.0, 0.0], [0.0, 0.0]])
    np.testing.assert_array_equal(cardinal_proms, [[-1.0, 0.0], [0.0, 4.0]])
    assert np.count_nonzero(d8_peaks) == 1
    assert not np.any(d8_saddles)
    assert len(np.unique(cardinal_peaks[cardinal_peaks > 0])) == 2
    # Both zero cells are possible passes, but only the first merge
    # plateau is the key saddle in the prominence tree.
    assert np.count_nonzero(cardinal_saddles) == 1


def test_compute_prominence_supports_unsigned_dem():
    dem = np.array([[3, 1, 2]], dtype=np.uint16)

    proms, peaks, saddles, _, _ = _compute_prominence_labels(dem)  # type: ignore

    assert proms.dtype == np.int64
    np.testing.assert_array_equal(proms, [[-1, 0, 1]])
    assert peaks.dtype == np.int32
    assert saddles.dtype == np.int32


@pytest.mark.parametrize(
    ("dtype", "high", "low", "subordinate"),
    [
        pytest.param(np.int8, 127, -128, 126, id="int8-extrema"),
        pytest.param(np.int16, 32767, -32768, 32766, id="int16-extrema"),
    ],
)
def test_compute_prominence_warns_for_signed_integer_result_overflow(
    dtype, high, low, subordinate
):
    dem = np.array([[high, low, subordinate]], dtype=dtype)

    with pytest.warns(RuntimeWarning, match="will overflow during conversion"):
        proms, *_ = compute_prominence(dem)

    assert proms.dtype == dtype


@pytest.mark.parametrize(
    "dem",
    [
        pytest.param(
            np.array([[2**24 + 1, 2**24, 2**24 + 2]], dtype=np.int64),
            id="int64-beyond-float32-exact-range",
        ),
        pytest.param(
            np.array(
                [[np.iinfo(np.uint32).max, 0, np.iinfo(np.uint32).max - 1]],
                dtype=np.uint32,
            ),
            id="large-uint32",
        ),
    ],
)
def test_compute_prominence_warns_when_large_integer_elevations_merge(dem):
    with pytest.warns(RuntimeWarning, match="merges distinct elevation values"):
        compute_prominence(dem)


def test_compute_prominence_warns_when_float64_elevations_merge():
    small_difference = 2.0**-25
    dem = np.array([[1.0 + small_difference, 1.0, 1.0 + 2.0**-23]], dtype=np.float64)

    with pytest.warns(RuntimeWarning, match="merges distinct elevation values"):
        compute_prominence(dem)


@pytest.mark.parametrize(
    ("dem", "exception"),
    [
        pytest.param(np.ones(3, dtype=np.float32), ValueError, id="one-dimensional"),
        pytest.param(np.empty((0, 3), dtype=np.float32), ValueError, id="empty"),
        pytest.param(np.ones((2, 2), dtype=np.complex64), TypeError, id="complex"),
        pytest.param(np.array([["high", "low"]]), TypeError, id="non-numeric"),
    ],
)
def test_compute_prominence_rejects_malformed_dems(dem, exception):
    with pytest.raises(exception):
        compute_prominence(dem)


def test_compute_prominence_rejects_mismatched_validity_mask():
    dem = np.ones((2, 3), dtype=np.float32)
    valids = np.ones((3, 2), dtype=bool)

    with pytest.raises(ValueError, match="Shapes for DEM and validity mask must match"):
        compute_prominence(dem, valids)


@pytest.mark.parametrize(
    ("offsets", "exception", "message"),
    [
        pytest.param(
            np.array([0, 1], dtype=np.int32),
            ValueError,
            "shape",
            id="one-dimensional",
        ),
        pytest.param(
            np.ones((2, 3), dtype=np.int32),
            ValueError,
            "shape",
            id="three-columns",
        ),
        pytest.param(
            np.array([[0.0, 0.0], [0.5, 1.0]]),
            TypeError,
            "integer dtype",
            id="fractional-offset",
        ),
        pytest.param(
            np.empty((0, 2), dtype=np.int32),
            ValueError,
            "at least one offset",
            id="no-offsets",
        ),
        pytest.param(
            np.array([[0, np.iinfo(np.int32).max + 1]], dtype=np.int64),
            ValueError,
            "representable as int32",
            id="outside-int32-range",
        ),
    ],
)
def test_compute_prominence_rejects_invalid_connectivity_offsets(
    offsets, exception, message
):
    dir_scheme = D8Directions()
    dir_scheme.offsets = offsets

    with pytest.raises(exception, match=message):
        compute_prominence(
            np.array([[3.0, 1.0, 2.0]], dtype=np.float32),
            dir_scheme=dir_scheme,
        )


def test_compute_prominence_feature_ids_are_deterministic_for_equal_plateaus():
    dem = np.array(
        [
            [5.0, 0.0, 5.0, 0.0, 6.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [5.0, 0.0, 5.0, 0.0, 6.0],
        ],
        dtype=np.float32,
    )
    variants = [
        np.array(dem, order="C"),
        np.array(dem, order="F"),
        dem[:, ::-1][:, ::-1],
    ]

    expected = compute_prominence(variants[0])
    for variant in variants[1:]:
        actual = compute_prominence(variant)
        for actual_array, expected_array in zip(actual, expected):
            np.testing.assert_array_equal(actual_array, expected_array)


def test_compute_prominence_returns_divide_tree_and_key_saddles():
    dem = np.array([[10.0, 6.0, 8.0, 4.0, 12.0]], dtype=np.float32)

    _, feats, feat_types, feat_ijs, key_saddles, feat_prnts = compute_prominence(dem)

    peak_10 = feats[0, 0]
    saddle_6 = feats[0, 1]
    peak_8 = feats[0, 2]
    saddle_4 = feats[0, 3]
    peak_12 = feats[0, 4]

    np.testing.assert_array_equal(feat_types, [1, 1, 1, 2, 2])
    np.testing.assert_array_equal(feats[feat_ijs[:, 0], feat_ijs[:, 1]], np.arange(5))
    assert feat_prnts[peak_10] == saddle_6
    assert feat_prnts[peak_8] == saddle_6
    assert feat_prnts[saddle_6] == saddle_4
    assert feat_prnts[peak_12] == saddle_4
    assert feat_prnts[saddle_4] == -1
    assert key_saddles[peak_8] == saddle_6
    assert key_saddles[peak_10] == saddle_4
    assert key_saddles[peak_12] == -1


def test_compute_prominence_tracks_copeaks_through_later_saddle():
    dem = np.array([[10.0, 6.0, 10.0, 4.0, 12.0]], dtype=np.float32)

    _, feats, feat_types, _, key_saddles, feat_prnts = compute_prominence(dem)

    peak_10a = feats[0, 0]
    saddle_6 = feats[0, 1]
    peak_10b = feats[0, 2]
    saddle_4 = feats[0, 3]
    peak_12 = feats[0, 4]

    assert feat_prnts[peak_10a] == saddle_6
    assert feat_prnts[peak_10b] == saddle_6
    assert feat_prnts[saddle_6] == saddle_4
    assert feat_prnts[peak_12] == saddle_4
    assert feat_prnts[saddle_4] == -1
    assert key_saddles[peak_10a] == saddle_4
    assert key_saddles[peak_10b] == saddle_4
    assert key_saddles[peak_12] == -1
    assert feat_types[saddle_6] == feat_types[saddle_4] == 2


@pytest.mark.parametrize(
    ("shape", "include_invalids", "seed"),
    [
        pytest.param((5, 5), False, 21, id="square-valid"),
        pytest.param((4, 7), True, 22, id="wide-masked"),
        pytest.param((7, 4), True, 23, id="tall-masked"),
        pytest.param((1, 11), True, 24, id="single-row-masked"),
    ],
)
def test_compute_prominence_feature_tree_invariants(shape, include_invalids, seed):
    rng = np.random.default_rng(seed)
    dem = rng.integers(-4, 10, size=shape).astype(np.float32)
    valids = np.ones(shape, dtype=bool)
    if include_invalids:
        valids.flat[::4] = False
        valids.flat[-1] = True

    proms, feats, feat_types, feat_ijs, key_saddles, feat_prnts = compute_prominence(
        dem, valids
    )

    nfeats = feat_types.size
    present = np.unique(feats[feats >= 0])
    np.testing.assert_array_equal(present, np.arange(nfeats, dtype=np.int32))
    assert np.all(feats[~valids] == -1)
    assert key_saddles.shape == feat_prnts.shape == (nfeats,)
    assert feat_ijs.shape == (nfeats, 2)
    assert np.all(np.isin(feat_types, [1, 2]))
    assert np.all((-1 <= key_saddles) & (key_saddles < nfeats))
    assert np.all((-1 <= feat_prnts) & (feat_prnts < nfeats))

    peaks = np.flatnonzero(feat_types == 1)
    saddles = np.flatnonzero(feat_types == 2)
    assert np.all(key_saddles[saddles] == -1)
    assert np.all(feat_types[feat_prnts[feat_prnts >= 0]] == 2)
    assert np.all((0 <= feat_ijs[:, 0]) & (feat_ijs[:, 0] < shape[0]))
    assert np.all((0 <= feat_ijs[:, 1]) & (feat_ijs[:, 1] < shape[1]))
    np.testing.assert_array_equal(
        feats[feat_ijs[:, 0], feat_ijs[:, 1]], np.arange(nfeats)
    )

    feat_zs = np.empty(nfeats, dtype=np.float32)
    for feat in range(nfeats):
        feat_cells = np.argwhere(feats == feat)
        zs = np.unique(dem[feats == feat])
        assert zs.size == 1
        feat_zs[feat] = zs[0]

        centroid = np.mean(feat_cells, axis=0)
        dist2 = np.sum((feat_cells - centroid) ** 2, axis=1)
        closest = feat_cells[dist2 == np.min(dist2)]
        linear_ids = closest[:, 0] + closest[:, 1] * shape[0]
        expected_ij = closest[np.argmin(linear_ids)]
        np.testing.assert_array_equal(feat_ijs[feat], expected_ij)

    for feat in range(nfeats):
        visited = set()
        ancestor = feat
        while feat_prnts[ancestor] >= 0:
            assert ancestor not in visited
            visited.add(ancestor)
            prnt = feat_prnts[ancestor]
            assert feat_zs[prnt] <= feat_zs[ancestor]
            ancestor = prnt

    for peak in peaks:
        peak_proms = np.unique(proms[feats == peak])
        assert peak_proms.size == 1
        key_saddle = key_saddles[peak]
        if key_saddle < 0:
            assert peak_proms[0] == -1
            continue

        assert feat_types[key_saddle] == 2
        ancestor = peak
        while ancestor >= 0 and ancestor != key_saddle:
            ancestor = feat_prnts[ancestor]
        assert ancestor == key_saddle
        assert peak_proms[0] == feat_zs[peak] - feat_zs[key_saddle]


def test_compute_prominence_representative_cell_tie_break():
    dem = np.array([[5.0, 1.0, 1.0, 4.0]], dtype=np.float32)

    _, feats, _, feat_ijs, _, _ = compute_prominence(dem)

    peak_5 = feats[0, 0]
    peak_4 = feats[0, 3]
    saddle = feats[0, 1]
    np.testing.assert_array_equal(feat_ijs[peak_5], [0, 0])
    np.testing.assert_array_equal(feat_ijs[peak_4], [0, 3])
    np.testing.assert_array_equal(feat_ijs[saddle], [0, 1])
