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
