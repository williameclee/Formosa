"""
Derives raster coordinates from digital elevation model metadata.

Last modified: 2026-08-23, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np
import rasterio.transform as rt
from numpy.typing import NDArray
from rasterio.transform import Affine


def transform2xy(
    transform: Affine, shape: tuple[int, int]
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Generates X and Y coordinate arrays from an affine transform.

    Parameters
    ----------
    transform : Affine
        Affine transformation mapping pixel coordinates to spatial
        coordinates.
    shape : tuple[int, int]
        Shape of the raster data as `(nrows, ncols)`.

    Returns
    -------
    x : NDArray[float]
        x-coordinates.
        - Shape: `shape`.
    y : NDArray[float]
        y-coordinates.
        - Shape: `shape`.
    """
    ii, jj = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    xx, yy = rt.xy(transform, jj, ii)
    xx = np.reshape(xx, (-1,)).reshape(shape)
    yy = np.reshape(yy, (-1,)).reshape(shape)

    return xx, yy
