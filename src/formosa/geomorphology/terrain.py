import numpy as np

import numpy.typing as npt


def compute_slope(
    dem: npt.NDArray[np.number],
    x: npt.NDArray[np.floating | np.integer] | None = None,
    y: npt.NDArray[np.floating | np.integer] | None = None,
    dx: int | float | None = None,
    dy: int | float | None = None,
) -> npt.NDArray[np.floating | np.integer]:
    if x is not None:
        dxx = np.gradient(x, axis=1)
    elif dx is not None:
        dxx = dx
    else:
        dxx = 1
    if y is not None:
        dyy = np.gradient(y, axis=0)
    elif dy is not None:
        dyy = dy
    else:
        dyy = 1

    slope_y, slope_x = np.gradient(dem)
    slope_x /= dxx
    slope_y /= dyy

    slope = np.sqrt(slope_x**2 + slope_y**2)
    return slope
