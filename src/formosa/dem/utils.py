import numpy as np
import rasterio.transform as rt
from rasterio.transform import Affine

import numpy.typing as npt


def transform2xy(
    transform: Affine, shape: tuple[int, int]
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Generate X, Y coordinate arrays from a rasterio affine transform and shape.

    Parameters
    ----------
    transform : Affine
        Affine transformation mapping pixel coordinates to spatial coordinates.
    shape : tuple[int, int]
        Shape of the raster data as (rows, columns).

    Returns
        -------
    x : ndarray[float]
        2D array of x-coordinates.
    y : ndarray[float]
        2D array of y-coordinates.
    """
    ii, jj = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    xx, yy = rt.xy(transform, jj, ii)
    xx = np.reshape(xx, (-1,)).reshape(shape)
    yy = np.reshape(yy, (-1,)).reshape(shape)

    return xx, yy
