"""
Prepares digital elevation models for drainage analysis in Python.

This module implements the internal Python backend called by the
public-facing drainage API and is not intended to be used directly.

Last modified: 2026-08-24, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np
from numpy.typing import NDArray

from formosa.utils import NpReal


def fill_depressions[Z: NpReal](
    dem: NDArray[Z], valids: NDArray[np.bool_]
) -> NDArray[Z]:
    """
    Fill D8 depressions using iterative reconstruction by erosion.

    Notes
    -----
    Deprecated and no longer called by the public functions.
    """
    recon = dem.copy()
    recon[valids] = np.inf

    for i in range(dem.shape[0]):
        for j in range(dem.shape[1]):
            if not valids[i, j]:
                continue
            i0, i1 = max(0, i - 1), min(dem.shape[0], i + 2)
            j0, j1 = max(0, j - 1), min(dem.shape[1], j + 2)
            is_outer_bdry = (
                i == 0 or i == dem.shape[0] - 1 or j == 0 or j == dem.shape[1] - 1
            )
            is_mask_bdry = np.any(~valids[i0:i1, j0:j1])
            if is_outer_bdry or is_mask_bdry:
                recon[i, j] = dem[i, j]

    while True:
        prev = recon.copy()
        for i in range(dem.shape[0]):
            for j in range(dem.shape[1]):
                if not valids[i, j]:
                    continue
                i0, i1 = max(0, i - 1), min(dem.shape[0], i + 2)
                j0, j1 = max(0, j - 1), min(dem.shape[1], j + 2)
                nabr_valids = valids[i0:i1, j0:j1]
                nabr_vals = prev[i0:i1, j0:j1][nabr_valids]
                recon[i, j] = max(dem[i, j], np.min(nabr_vals))
        if np.array_equal(recon, prev):
            return recon
