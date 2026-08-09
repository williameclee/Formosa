"""
Operations on the watershed/drainage basin raster.

Last modified: 2026-08-09, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np

from formosa.utils import Backend, raise_fortran_error
from formosa.geomorphology.drainage.directions import D8Directions
from formosa.geomorphology._native import drainage_watersheds as basins_f
import formosa.geomorphology.drainage._backends.watersheds_py as watersheds_py

from typing import Optional
import numpy.typing as npt


def label_watersheds(
    dirs: npt.NDArray[np.integer],
    dir_scheme: D8Directions = D8Directions(),
    valids: Optional[npt.NDArray[np.bool_]] = None,
    backend: Backend = "fortran",
) -> npt.NDArray[np.int32]:
    """
    Finds and labels watersheds in a DEM based on flow direction.

    Parameters
    ----------
    dirs : NDArray[int]
        A 2D array representing the flow direction for each cell.
    dir_scheme : D8Directions, optional
        An instance of `D8Directions` defining the flow direction scheme.
        Default is `D8Directions()`.
    valids : NDArray[bool], optional
        A boolean mask array indicating valid cells in the flow direction grid.
        If `None`, all non-NaN cells in flowdirs are considered valid.
        Default is `None`.
    backend : {'fortran', 'python'}, optional
        Backend to use for computation.
        `'fortran'` uses the FORTRAN extension for performance,
        while `'python'` uses a pure Python implementation.
        Default backend is `'fortran'`.

    Returns
    -------
    watersheds : NDArray[int32]
        A 2D array where each watershed is labeled with a unique integer.
    """
    match backend:
        case "python":
            watersheds = watersheds_py.label_watersheds(
                dirs=dirs,
                dir_scheme=dir_scheme,
                valids=valids,
            )
        case "fortran":
            if valids is None:
                valids = np.ones(dirs.shape, dtype=bool)
            elif isinstance(valids, np.ndarray):
                assert (
                    valids.shape == dirs.shape
                ), f"Shape for flow direction ({dirs.shape}) and valid mask ({valids.shape}) do not match."
                valids = valids.astype(bool, copy=False) & (~np.isnan(dirs))
                dirs = np.where(valids, dirs, np.nan)
            else:
                raise TypeError(
                    f"Valid mask must be a NumPy array (got {type(valids)})."
                )

            watersheds, err_code = basins_f.label_watersheds(
                dirs.astype(np.uint8, order="F"),
                valids.astype(bool, order="F"),
                dir_scheme.offsets.astype(np.int32, order="F"),
                dir_scheme.codes.astype(np.uint8, order="F"),
            )
            raise_fortran_error("label_watersheds", err_code)
    return watersheds.astype(np.int32, order="F")
