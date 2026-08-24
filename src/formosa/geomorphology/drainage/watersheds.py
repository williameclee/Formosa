"""
Identifies and labels watersheds from raster flow directions.

Created: 2026-08-01, En-Chi Lee (williameclee@gmail.com)
Last modified: 2026-08-24, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np
from numpy.typing import NDArray

import formosa.geomorphology.drainage._backends.watersheds_py as wsheds_py
from formosa.geomorphology._native import drainage_watersheds as basins_f
from formosa.geomorphology.drainage.directions import DirectionEncoding
from formosa.geomorphology.raster_validation import (
    validate_format_dir_encoding,
    validate_format_flowdirs,
    validate_format_valids,
)
from formosa.utils import Backend, NpFlowDir, raise_fortran_error


def label_watersheds(
    dirs: NDArray[NpFlowDir],
    dir_enc: DirectionEncoding | None = None,
    valids: NDArray[np.bool_] | None = None,
    backend: Backend = "fortran",
) -> NDArray[np.int32]:
    """
    Finds and labels watersheds in a DEM based on flow direction.

    Parameters
    ----------
    dirs : NDArray[uint8]
        Flow direction raster.
        - Expected shape: `(nrows, ncols)`.
    dir_enc : DirectionEncoding, optional
        Flow direction encoding scheme.
        - Default scheme is `D8DirectionEncoding()`.
    valids : NDArray[bool], optional
        Boolean mask indicating valid cells in the flow direction
        grid.
        If `None`, all cells are considered valid.
        - Expected shape: `(nrows, ncols)`, same as `dirs`.
        - Default mask is `None`.
    backend : {'fortran', 'python'}, optional
        Backend to use for computation.
        `'fortran'` uses the Fortran extension for performance,
        while `'python'` uses a pure Python implementation.
        - Default backend is `'fortran'`.

    Returns
    -------
    ws : NDArray[int32]
        Watershed labels where each watershed is labelled with a
        unique integer.
        - Shape: `(nrows, ncols)`, same as `dirs`.
    """
    dirs = validate_format_flowdirs(dirs)
    dir_enc = validate_format_dir_encoding(dir_enc)
    valids = validate_format_valids(valids, dirs, "flow direction raster")

    match backend:
        case "python":
            ws = wsheds_py.label_watersheds(dirs, dir_enc, valids=valids)
        case "fortran":
            ws, err_code = basins_f.label_watersheds(
                dirs.astype(np.uint8, order="F"),
                valids.astype(bool, order="F"),
                dir_enc.offsets.astype(np.int32, order="F"),
                dir_enc.codes.astype(np.uint8, order="F"),
            )
            raise_fortran_error("label_watersheds", err_code)
    return ws.astype(np.int32, order="F")
