# Last modified
#   2026-06-11, En-Chi Lee (williameclee@gmail.com)
#     - Moved Python backend implementations to this file
#     - Removed redundant NaN checks against integer arrays
#     - Standardised variable, argument, and function names
#   2026-07-02, En-Chi Lee (williameclee@gmail.com)
#     - Updated indegree algorithm
#   2026-07-09, En-Chi Lee (williameclee@gmail.com)
#     - Added better validity check in `count_indegree`
#   2026-08-03, En-Chi Lee (williameclee@gmail.com)
#     - Implemented Python backend for function
#       `find_acyclic_flowdirs`.

import numpy as np
from collections import deque

from formosa.geomorphology.drainage.directions import D8Directions
from formosa.geomorphology.drainage.neighbours import (
    get_neighbour_values,
    compute_downstream_indices,
)

from typing import Optional
import numpy.typing as npt


def compute_flowdir_simple(
    dem: npt.NDArray[np.number],
    dir_scheme: D8Directions = D8Directions(),
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.bool_]]:
    neighbours, codes, _ = get_neighbour_values(
        dem, dir_scheme=dir_scheme, include_self=True, pad_value=np.max(dem) + 1
    )
    flow2self_code = np.where(np.all(dir_scheme.offsets == [0, 0], axis=1))[0][0]
    flowdirs = np.full(dem.shape, flow2self_code, dtype=np.int32)
    # find where not all neighbours are nan
    valid_mask = ~np.all(np.isnan(neighbours), axis=0)
    flowdirs[valid_mask] = np.nanargmin(neighbours[:, valid_mask], axis=0)

    flowdirs = codes[flowdirs].astype(np.int32)
    is_flat = flowdirs == 0
    return flowdirs, is_flat


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


def count_indegree(
    dirs: npt.NDArray[np.integer],
    dir_scheme: D8Directions = D8Directions(),
    valids: Optional[npt.NDArray[np.bool_]] = None,
) -> npt.NDArray[np.int8]:
    if valids is None:
        valids = np.ones(dirs.shape, dtype=bool)
    indegs = np.zeros(dirs.shape, dtype=np.int8)
    dsi, dsj, _, ds_valids = compute_downstream_indices(
        dirs, dir_scheme=dir_scheme, valids=valids, check=False, return_flat_index=False
    )

    for i in range(dirs.shape[0]):
        for j in range(dirs.shape[1]):
            if not valids[i, j]:
                continue
            if not ds_valids[i, j]:
                continue
            elif (dsi[i, j] == i) and (dsj[i, j] == j):  # skip self-loop
                continue
            indegs[dsi[i, j], dsj[i, j]] += 1
    # TODO: Find out why is there overflow here?
    return indegs


def find_acyclic_flowdirs(
    dirs: npt.NDArray[np.integer],
    indegs: npt.NDArray[np.integer],
    valids: npt.NDArray[np.bool_],
    dir_scheme: D8Directions = D8Directions(),
) -> npt.NDArray[np.bool_]:
    """Finds valid cells that do not belong to a directed flow cycle."""
    remaining_indegs = np.asarray(indegs, dtype=np.int8).copy()
    acyclics = np.zeros(valids.shape, dtype=bool)
    queue = deque(map(tuple, np.argwhere(valids & (remaining_indegs == 0))))

    dsi, dsj, _, ds_inbounds = compute_downstream_indices(
        dirs,
        dir_scheme=dir_scheme,
        check=False,
        return_flat_index=False,
        oob_is_okay=True,
    )
    ds_valids = np.zeros(valids.shape, dtype=bool)
    ds_valids[ds_inbounds] = valids[dsi[ds_inbounds], dsj[ds_inbounds]]
    di, dj = dir_scheme.code2d8offset(dirs)
    has_valid_ds = valids & ds_valids & ((di != 0) | (dj != 0))

    while queue:
        i, j = queue.popleft()
        if acyclics[i, j]:
            continue
        acyclics[i, j] = True
        if not has_valid_ds[i, j]:
            continue

        ni = dsi[i, j]
        nj = dsj[i, j]
        remaining_indegs[ni, nj] -= 1
        if remaining_indegs[ni, nj] == 0:
            queue.append((ni, nj))

    return acyclics
