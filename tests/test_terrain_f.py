"""
Tests terrain metrics using the FORTRAN backend.

Created: 2026-08-19, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np
import pytest

from formosa.geomorphology.terrain import calculate_isolation


def _brute_force_isolation(
    dem: np.ndarray, valids: np.ndarray, dx: float, dy: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns exact isolation distances and whether an ILP exists.
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

    return isos, has_ilp


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

    expected_isos, expected_has_ilp = _brute_force_isolation(dem, valids, dx, dy)
    isos, ilpis, ilpjs = calculate_isolation(dem, valids, dx=dx, dy=dy)

    np.testing.assert_allclose(isos, expected_isos, rtol=1e-6, atol=1e-6)
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

    isos, ilpis, ilpjs = calculate_isolation(dem, dx=10.0, dy=1.0)

    assert isos[2, 2] == pytest.approx(2.0)
    assert (ilpis[2, 2], ilpjs[2, 2]) == (4, 2)


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

    isos, ilpis, ilpjs = calculate_isolation(dem)

    assert isos[1, 1] == pytest.approx(np.sqrt(5.0))
    assert (ilpis[1, 1], ilpjs[1, 1]) == (2, 3)
    assert isos[2, 2] == pytest.approx(1.0)
    assert (ilpis[2, 2], ilpjs[2, 2]) == (2, 3)
    assert isos[2, 3] == 0.0
    assert (ilpis[2, 3], ilpjs[2, 3]) == (-1, -1)


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

    isos, ilpis, ilpjs = calculate_isolation(dem, valids)

    assert isos[1, 1] == pytest.approx(np.sqrt(2.0))
    assert (ilpis[1, 1], ilpjs[1, 1]) == (2, 2)
    assert isos[0, 0] == 0.0
    assert (ilpis[0, 0], ilpjs[0, 0]) == (-1, -1)
    assert isos[1, 2] == 0.0
    assert (ilpis[1, 2], ilpjs[1, 2]) == (-1, -1)


@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (5, 4)])
def test_all_invalid_cells_have_no_isolation_limit_point(shape):
    dem = np.ones(shape, dtype=np.float32)
    valids = np.zeros(shape, dtype=bool)

    isos, ilpis, ilpjs = calculate_isolation(dem, valids)

    assert np.all(isos == 0.0)
    assert np.all(ilpis == -1)
    assert np.all(ilpjs == -1)


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
        calculate_isolation(np.ones((2, 2), dtype=np.float32), **kwargs)
