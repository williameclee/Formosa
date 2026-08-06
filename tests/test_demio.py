import numpy as np
import pytest

from formosa.dem import DEMGrid, read_dem


def test_read_hgt_and_construct_demgrid(tmp_path):
    # The Rasterio SRTMHGT driver recognizes the standard 1201- or 3601-cell
    # tile sizes and derives the geographic extent from the filename.
    elevations = np.arange(1201 * 1201, dtype=np.int64).reshape(1201, 1201)
    elevations = (elevations % 3000).astype(">i2")
    elevations[0, 0] = -32768
    path = tmp_path / "N25E121.hgt"
    elevations.tofile(path)

    dem, x, y, transform = read_dem(path)

    assert dem.shape == (1201, 1201)
    assert np.isnan(dem[0, 0])
    assert dem[0, 1] == elevations[0, 1]
    assert transform.a == pytest.approx(1 / 1200)
    assert transform.e == pytest.approx(-1 / 1200)
    # HGT elevations are posts on the integer-degree tile boundaries.
    assert x[0, 0] == pytest.approx(121)
    assert y[0, 0] == pytest.approx(26)

    grid = DEMGrid(path)
    assert grid.shape == (1201, 1201)
    assert not grid.valid[0, 0]
    assert grid.dem[0, 1] == elevations[0, 1]


def test_demgrid_invalidate_ocean_basins_is_chainable_and_updates_sea_mask():
    dem = np.full((7, 9), 5.0, dtype=np.float32)
    dem[0:2, 0:3] = 0.0  # Six-cell boundary basin.
    dem[5:7, 7:9] = 0.0  # Four-cell boundary basin.
    dem[3, 3:6] = 0.0  # Enclosed low basin.
    original = dem.copy()
    x, y = np.meshgrid(np.arange(dem.shape[1]), np.arange(dem.shape[0]))
    grid = DEMGrid(dem.copy(), x=x, y=y)

    result = grid.invalidate_ocean_basins(min_size=6)

    assert result is grid
    assert not np.any(grid.valid[0:2, 0:3])
    assert np.all(grid.valid[5:7, 7:9])
    assert np.all(grid.valid[3, 3:6])
    np.testing.assert_array_equal(grid.ocean_mask, ~grid.valid)
    np.testing.assert_array_equal(grid.dem, original)


def test_demgrid_detect_ocean_uses_boundary_basins_and_size_threshold():
    dem = np.full((7, 9), 5.0, dtype=np.float32)
    dem[0:2, 0:3] = 0.0  # Six-cell boundary basin.
    dem[5:7, 7:9] = 0.0  # Four-cell boundary basin.
    dem[3, 3:6] = 0.0  # Enclosed low basin.
    x, y = np.meshgrid(np.arange(dem.shape[1]), np.arange(dem.shape[0]))

    grid = DEMGrid(
        dem.copy(),
        x=x,
        y=y,
        detect_ocean=True,
        min_ocean_size=6,
    )

    assert not np.any(grid.valid[0:2, 0:3])
    assert np.all(grid.valid[5:7, 7:9])
    assert np.all(grid.valid[3, 3:6])
    np.testing.assert_array_equal(grid.ocean_mask, ~grid.valid)


def test_demgrid_detect_ocean_true_means_zero_elevation():
    dem = np.full((3, 3), 5.0, dtype=np.float32)
    dem[0, 0] = 0.5
    x, y = np.meshgrid(np.arange(dem.shape[1]), np.arange(dem.shape[0]))

    grid = DEMGrid(dem, x=x, y=y, detect_ocean=True)

    assert grid.valid[0, 0]
    assert grid.ocean_threshold == 0
