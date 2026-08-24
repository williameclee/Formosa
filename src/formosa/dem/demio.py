"""
Reads digital elevation model data from raster files on disk.

Last modified: 2026-08-24, En-Chi Lee (williameclee@gmail.com)
"""

from pathlib import Path

import numpy as np
import rasterio
from numpy.typing import NDArray

from formosa.dem.utils import transform2xy
from formosa.geomorphology.raster_validation import (
    validate_format_dem,
)
from formosa.utils import NpCoords, NpReal


def read_dem(
    raster_path: Path | str,
    band: int = 1,
    nan_value: float = np.nan,
) -> tuple[
    NDArray[NpReal],
    NDArray[NpCoords],
    NDArray[NpCoords],
    rasterio.Affine,
]:
    """
    Reads a DEM from a raster file.

    GeoTIFF and SRTM `.hgt` tiles are supported. HGT georeferencing
    is inferred from the standard tile name (for example,
    `N25E121.hgt`), so HGT files must retain that naming convention.

    Parameters
    ----------
    raster_path : Path | str
        Path to a supported raster DEM. For HGT input, the filename
        must identify the tile's southwest corner.
    band : int, optional
        Band number to read from the GeoTIFF file.
        - Default band is `1`.
    nan_value : float, optional
        Value to use for no-data pixels.
        - Default value is `np.nan`.

    Returns
    -------
    dem : NDArray[number]
        Elevation values.
        - Shape: `(nrows, ncols)`.
    x : NDArray[float]
        x-coordinates corresponding to `dem`.
        - Shape: `(nrows, ncols)`, same as `dem`.
    y : NDArray[float]
        y-coordinates corresponding to `dem`.
        - Shape: `(nrows, ncols)`, same as `dem`.
    transform : rasterio.Affine
        Affine transformation mapping pixel coordinates to spatial
        coordinates.
    """

    with rasterio.open(raster_path) as src:
        # Read the DEM band (assuming band 1 is elevation)
        dem = src.read(band)

        # Check Z is a valid numpy array
        dem = validate_format_dem(dem)

        # Handle no-data values
        if src.nodata is not None:
            dem = np.where(dem == src.nodata, nan_value, dem)

        dem = np.asarray(dem, order="F")
        # Make coordinate arrays
        transform = (
            src.transform if src.transform is not None else rasterio.Affine.identity()
        )
        x, y = transform2xy(transform, dem.shape)

    return dem, x, y, transform
