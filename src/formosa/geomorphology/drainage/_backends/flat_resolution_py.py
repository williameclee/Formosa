"""
Resolves flats in digital elevation models using the Python backend.

This module implements internal routines called by the public-facing
drainage API and is not intended to be used directly.

Last modified: 2026-08-10, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np

from formosa.geomorphology.drainage.directions import D8Directions
from formosa.geomorphology.drainage.neighbours import get_neighbour_values

import numpy.typing as npt


def compute_masked_flowdir(
    z: npt.NDArray[np.integer | np.floating],
    labels: npt.NDArray[np.integer],
    dir_scheme: D8Directions = D8Directions(),
) -> npt.NDArray[np.integer]:
    neighbours, codes, _ = get_neighbour_values(
        z,
        dir_scheme=dir_scheme,
        include_self=True,
        pad_value=z.max() + 1,
    )
    neighbour_labels, _, _ = get_neighbour_values(
        labels, dir_scheme=dir_scheme, include_self=True, pad_value=-1
    )
    # Mask neighbours that are not in the same flat
    neighbours = np.where(
        neighbour_labels != labels[np.newaxis, :, :], np.inf, neighbours
    )
    min_indices = np.argmin(neighbours, axis=0)
    flowdirs = codes[min_indices]
    flowdirs[labels == 0] = 0

    return flowdirs


def find_flat_edges(
    dem: npt.NDArray[np.number],
    dirs: npt.NDArray[np.integer],
    dir_scheme=D8Directions(),
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    neighbours, _, _ = get_neighbour_values(
        dem,
        dir_scheme=dir_scheme,
        include_self=False,
        pad_value=np.min(dem) - 1,  # since is_high_edge
    )
    neighbour_flowdirs, _, _ = get_neighbour_values(
        dirs, dir_scheme=dir_scheme, include_self=False, pad_value=-1
    )

    is_high_edge: npt.NDArray[np.bool_] = (dirs == 0) & np.any(dem < neighbours, axis=0)
    is_low_edge: npt.NDArray[np.bool_] = (dirs != 0) & (
        np.any((neighbour_flowdirs == 0) & (dem == neighbours), axis=0)
    )

    return is_low_edge, is_high_edge
