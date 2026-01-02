from pathlib import Path
import numpy as np
import rasterio
import rasterio.transform as rtransform

import numpy.typing as npt


def read_dem(
    tiff_path: Path | str,
    band: int = 1,
    nan_value: float = np.nan,
) -> tuple[
    npt.NDArray[np.floating | np.integer],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    rasterio.Affine,
]:
    """
    Read a DEM from a GeoTIFF file.

    Parameters
    ----------
    tiff_path : Path | str
        Path to the GeoTIFF file containing the DEM data.
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

    with rasterio.open(tiff_path) as src:
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

        # Make coordinate arrays
        transform = (
            src.transform if src.transform is not None else rasterio.Affine.identity()
        )
        nrows, ncols = Z.shape
        ii, jj = np.meshgrid(np.arange(ncols), np.arange(nrows))
        X, Y = rtransform.xy(transform, jj, ii)

    return Z, X, Y, transform
