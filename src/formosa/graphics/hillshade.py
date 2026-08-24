"""
Renders shaded relief from two-dimensional elevation arrays.

Last modified: 2026-08-23, En-Chi Lee (williameclee@gmail.com)
"""

from typing import Literal

import numpy as np
from numpy.typing import NDArray


def hillshade(
    dem: NDArray[np.integer | np.floating],
    az: float,
    al: float,
    method: Literal["clamped", "hard", "half", "soft", "lambert"] = "hard",
    zfactor: float = 1,
) -> NDArray[np.floating]:
    """
    Computes shaded relief intensity from a 2D elevation grid.

    Parameters
    ----------
    dem : NDArray[number]
        Digital elevation model raster.
        - Expected shape: `(nrows, ncols)`.
    az : float
        Light source azimuth angle in degrees clockwise from North.
    al : float
        Light source altitude angle in degrees above the horizon.
    method : {"clamped", "hard", "half", "soft", "lambert"}, optional
        Shading response curve to apply.
        - Default method is `"hard"`.
    zfactor : float, optional
        Vertical exaggeration factor.
        - Default factor is `1`.

    Returns
    -------
    intst : NDArray[float]
        Hillshade intensity values in `[0, 1]`.
        - Shape: `(nrows, ncols)`, same as `dem`.
    """
    dem = dem * zfactor
    az = 180 - az
    lightvec: NDArray[np.floating] = np.array(
        [
            np.cos(np.deg2rad(al)) * np.cos(np.deg2rad(az)),
            np.cos(np.deg2rad(al)) * np.sin(np.deg2rad(az)),
            np.sin(np.deg2rad(al)),
        ]
    )

    dx, dy = np.gradient(dem)
    nvec: NDArray[np.integer | np.floating] = np.dstack((-dx, -dy, np.ones_like(dem)))
    nvec /= np.linalg.norm(nvec, axis=2, keepdims=True)
    intst: NDArray[np.floating] = np.sum(nvec * lightvec, axis=2)

    match method.lower():
        case "clamped" | "hard":
            intst = np.clip(intst, 0, 1)
        case "half":
            intst = (intst + 1) / 2
        case "soft" | "lambert":
            intst = (intst + 1) / 2
            intst = intst**2
        case _:
            raise ValueError(
                "Hillshade method must be 'hard' ('clamped'), 'half', or 'soft' ('Lambert'), "
                + f"but got '{method}' instead."
            )
    return intst
