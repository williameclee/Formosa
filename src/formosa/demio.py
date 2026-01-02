from pathlib import Path
import numpy as np
import rasterio


def read_dem(
    tiff_path: Path | str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, rasterio.Affine]:
    with rasterio.open(tiff_path) as src:
        # Read the DEM band (assuming band 1 is elevation)
        Z = src.read(1)

        # Handle no-data values
        if src.nodata is not None:
            Z = np.where(Z == src.nodata, np.nan, Z)

        # Get transform info (maps pixel to real-world coords)
        transform = (
            src.transform if src.transform is not None else rasterio.Affine.identity()
        )
        nrows, ncols = Z.shape

        # Get 1D coordinates for columns (x) and rows (y)
        x_coords = np.arange(ncols) * transform.a + transform.c
        y_coords = np.arange(nrows) * transform.e + transform.f

        # Make 2D meshgrid
        X, Y = np.meshgrid(x_coords, y_coords)

    return Z, X, Y, transform
