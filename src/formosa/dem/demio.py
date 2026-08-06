# Last modified
#   2026-08-06, En-Chi Lee (williameclee@gmail.com)
#     - Added support for `.hgt` files

from pathlib import Path
import numpy as np
import rasterio
from formosa.dem.utils import transform2xy

import numpy.typing as npt


def read_dem(
    raster_path: Path | str,
    band: int = 1,
    nan_value: float = np.nan,
) -> tuple[
    npt.NDArray[np.floating | np.integer],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    rasterio.Affine,
]:
    """
    Read a DEM from a raster file.

    GeoTIFF and SRTM `.hgt` tiles are supported. HGT georeferencing is
    inferred from the standard tile name (for example, `N25E121.hgt`), so
    HGT files must retain that naming convention.

    Parameters
    ----------
    raster_path : Path | str
        Path to a supported raster DEM. For HGT input, the filename must
        identify the tile's southwest corner.
    band : int, optional
        The band number to read from the GeoTIFF file (default is 1).
    nan_value : float, optional
        Value to use for no-data pixels (default is np.nan).

    Returns
    -------
    Z : ndarray[floating | integer]
        2D array of elevation values.
    X : ndarray[floating]
        2D array of x-coordinates corresponding to Z.
    Y : ndarray[floating]
        2D array of y-coordinates corresponding to Z.
    transform : rasterio.Affine
        Affine transformation mapping pixel coordinates to spatial coordinates.
    """

    with rasterio.open(raster_path) as src:
        # Read the DEM band (assuming band 1 is elevation)
        Z = src.read(band)

        # Check Z is a valid numpy array
        if not isinstance(Z, np.ndarray):
            raise ValueError(
                f"DEM data could not be read as a numpy array (type {type(Z)})."
            )
        # Check Z is 2D
        elif Z.ndim != 2:
            raise ValueError(
                f"DEM data must be a 2D array, got {Z.ndim}D array (shape {Z.shape})."
            )
        # Check number type is float or int
        elif not np.issubdtype(Z.dtype, np.floating) and not np.issubdtype(
            Z.dtype, np.integer
        ):
            raise ValueError(
                f"DEM data array must be of float or integer type (got {Z.dtype})."
            )

        # Handle no-data values
        if src.nodata is not None:
            Z = np.where(Z == src.nodata, nan_value, Z)

        Z = np.asarray(Z, order="F")
        # Make coordinate arrays
        transform = (
            src.transform if src.transform is not None else rasterio.Affine.identity()
        )
        X, Y = transform2xy(transform, Z.shape)

    return Z, X, Y, transform
