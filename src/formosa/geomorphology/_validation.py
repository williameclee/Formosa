import numpy as np

from formosa.utils import NpReal

from typing import Optional
from numpy.typing import NDArray


def _validate_format_dem(dem: NDArray[NpReal]) -> NDArray[NpReal]:
    dem = np.asarray(dem)
    if dem.ndim != 2 or 0 in dem.shape:
        raise ValueError(
            "DEM must be a non-empty 2D array, " + f"but received shape {dem.shape}."
        )
    if not np.issubdtype(dem.dtype, np.number):
        raise TypeError("DEM must have a numeric dtype, " + f"but got {dem.dtype}.")
    if np.issubdtype(dem.dtype, np.complexfloating):
        raise TypeError(
            "DEM must contain real-valued elevations, " + f"but got type {dem.dtype}."
        )
    return dem


def _validate_format_valids(
    valids: Optional[NDArray[np.bool_]], dem: NDArray[NpReal]
) -> NDArray[np.bool_]:
    finite = np.isfinite(dem)
    if valids is None:
        return finite
    valids = np.asarray(valids, dtype=bool)
    if valids.shape != dem.shape:
        raise ValueError(
            "Shapes for DEM and validity mask must match, "
            f"but got shapes {dem.shape} and {valids.shape}, respectively."
        )
    valids = valids & finite
    return valids  # type: ignore
