"""
Identifies and labels watersheds from raster flow directions.

Last modified: 2026-08-23, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np

from formosa.geomorphology.drainage.directions import D8Directions
from formosa.geomorphology._native import drainage_watersheds as basins_f
import formosa.geomorphology.drainage._backends.watersheds_py as watersheds_py
from formosa.utils import Backend, raise_fortran_error
from formosa.utils import NpFlowDir
from formosa.geomorphology._validation import (
    validate_format_valids,
    validate_format_flowdirs,
)

from typing import Optional
from numpy.typing import NDArray


def label_watersheds(
    dirs: NDArray[NpFlowDir],
    dir_scheme: D8Directions = D8Directions(),
    valids: Optional[NDArray[np.bool_]] = None,
    backend: Backend = "fortran",
) -> NDArray[np.int32]:
    """
    Finds and labels watersheds in a DEM based on flow direction.

    Parameters
    ----------
    dirs : NDArray[uint8]
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
        `'fortran'` uses the Fortran extension for performance,
        while `'python'` uses a pure Python implementation.
        Default backend is `'fortran'`.

    Returns
    -------
    watersheds : NDArray[int32]
        A 2D array where each watershed is labelled with a unique integer.
    """
    match backend:
        case "python":
            watersheds = watersheds_py.label_watersheds(
                dirs=dirs,
                dir_scheme=dir_scheme,
                valids=valids,
            )
        case "fortran":
            dirs = validate_format_flowdirs(dirs)
            valids = validate_format_valids(valids, dirs, "flow direction raster")
            watersheds, err_code = basins_f.label_watersheds(
                dirs.astype(np.uint8, order="F"),
                valids.astype(bool, order="F"),
                dir_scheme.offsets.astype(np.int32, order="F"),
                dir_scheme.codes.astype(np.uint8, order="F"),
            )
            raise_fortran_error("label_watersheds", err_code)
    return watersheds.astype(np.int32, order="F")
