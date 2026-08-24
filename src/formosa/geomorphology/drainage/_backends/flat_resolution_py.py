"""
Resolves flats in digital elevation models using the Python backend.

This module implements internal routines called by the public-facing
drainage API and is not intended to be used directly.

Last modified: 2026-08-24, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np
from numpy.typing import NDArray

from formosa.geomorphology.drainage.directions import DirectionEncoding
from formosa.geomorphology.drainage.neighbours import get_neighbour_values
from formosa.geomorphology.raster_validation import validate_format_dir_encoding
from formosa.utils import NpFlowDir, NpReal


def compute_masked_flowdir(
    z: NDArray[NpReal],
    labels: NDArray[np.integer],
    dir_enc: DirectionEncoding | None = None,
) -> NDArray[NpFlowDir]:
    dir_enc = validate_format_dir_encoding(dir_enc)

    nabrs, codes, _ = get_neighbour_values(
        z, dir_enc, include_self=True, pad_val=z.max() + 1
    )
    nabr_labels, _, _ = get_neighbour_values(
        labels, dir_enc, include_self=True, pad_val=-1
    )
    # Mask neighbours that are not in the same flat
    nabrs = np.where(nabr_labels != labels[np.newaxis, :, :], np.inf, nabrs)
    min_indices = np.argmin(nabrs, axis=0)
    flowdirs = codes[min_indices]
    flowdirs[labels == 0] = 0

    return flowdirs.astype(NpFlowDir)


def find_flat_edges(
    dem: NDArray[NpReal],
    dirs: NDArray[NpFlowDir],
    dir_enc: DirectionEncoding | None = None,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    dir_enc = validate_format_dir_encoding(dir_enc)

    nabrs, _, _ = get_neighbour_values(
        dem, dir_enc, include_self=False, pad_val=np.min(dem) - 1
    )
    nabr_dirs, _, _ = get_neighbour_values(
        dirs, dir_enc, include_self=False, pad_val=-1
    )

    is_high_edge: NDArray[np.bool_] = (dirs == 0) & np.any(dem < nabrs, axis=0)
    is_low_edge: NDArray[np.bool_] = (dirs != 0) & (
        np.any((nabr_dirs == 0) & (dem == nabrs), axis=0)
    )

    return is_low_edge, is_high_edge
