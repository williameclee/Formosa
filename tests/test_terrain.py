"""Tests terrain operations that do not require a native backend result."""

import numpy as np
import pytest

from formosa.geomorphology.terrain import compute_slope


def test_compute_slope_accepts_dem_as_required_first_argument():
    rows, cols = np.indices((3, 4))
    dem = 2.0 * cols + rows

    slope = compute_slope(dem)

    np.testing.assert_allclose(slope, np.sqrt(5.0))


def test_compute_slope_uses_coordinate_rasters_for_local_spacing():
    rows, cols = np.indices((3, 4))
    x = cols * 2.0
    y = rows * 4.0
    dem = 3.0 * x + 4.0 * y

    slope = compute_slope(dem, x=x, y=y)

    np.testing.assert_allclose(slope, 5.0)


@pytest.mark.parametrize("axis", ["x", "y"])
def test_compute_slope_rejects_coordinate_shape_mismatch(axis):
    coordinate = np.zeros((2, 3))
    kwargs = {axis: coordinate}

    with pytest.raises(ValueError, match="must match"):
        compute_slope(np.zeros((3, 3)), **kwargs)
