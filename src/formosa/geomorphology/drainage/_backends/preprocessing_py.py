# Last modified
#   2026-08-07 [PR 33], En-Chi Lee (williameclee@gmail.com)
#     - Added reconstruction-by-erosion backend for depression
#       filling.


import numpy as np

import numpy.typing as npt
from typing import Optional


def fill_depressions(
    dem: npt.NDArray[np.floating],
    valids: Optional[npt.NDArray[np.bool_]] = None,
) -> npt.NDArray[np.floating]:
    """
    Fill D8 depressions using iterative reconstruction by erosion.

    Notes
    -----
    Deprecated and no longer called by the public functions.
    """
    if valids is None:
        valids = np.ones(dem.shape, dtype=bool)

    reconstructed = dem.copy()
    reconstructed[valids] = np.inf

    for i in range(dem.shape[0]):
        for j in range(dem.shape[1]):
            if not valids[i, j]:
                continue
            i0, i1 = max(0, i - 1), min(dem.shape[0], i + 2)
            j0, j1 = max(0, j - 1), min(dem.shape[1], j + 2)
            is_outer_boundary = (
                i == 0 or i == dem.shape[0] - 1 or j == 0 or j == dem.shape[1] - 1
            )
            is_mask_boundary = np.any(~valids[i0:i1, j0:j1])
            if is_outer_boundary or is_mask_boundary:
                reconstructed[i, j] = dem[i, j]

    while True:
        previous = reconstructed.copy()
        for i in range(dem.shape[0]):
            for j in range(dem.shape[1]):
                if not valids[i, j]:
                    continue
                i0, i1 = max(0, i - 1), min(dem.shape[0], i + 2)
                j0, j1 = max(0, j - 1), min(dem.shape[1], j + 2)
                neighbour_valids = valids[i0:i1, j0:j1]
                neighbour_values = previous[i0:i1, j0:j1][neighbour_valids]
                reconstructed[i, j] = max(dem[i, j], np.min(neighbour_values))
        if np.array_equal(reconstructed, previous):
            return reconstructed
