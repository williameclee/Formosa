"""
Computes raster flow directions using the Python backend.

This module also provides raster-level analyses of the resulting
flow field; flow-graph operations are implemented in the network
package. These internal routines are called by the public-facing
drainage API.

Last modified: 2026-08-24, En-Chi Lee (williameclee@gmail.com)
"""

from collections import deque

import numpy as np
from numpy.typing import NDArray

from formosa.geomorphology.drainage.directions import DirectionEncoding
from formosa.geomorphology.drainage.neighbours import (
    compute_downstream_indices,
    get_neighbour_values,
)
from formosa.geomorphology.raster_validation import (
    validate_format_dir_encoding,
    validate_format_valids,
)
from formosa.utils import NpFlowDir, NpReal


def compute_flowdir_simple(
    dem: NDArray[NpReal],
    dir_enc: DirectionEncoding | None = None,
) -> tuple[NDArray[NpFlowDir], NDArray[np.bool_]]:
    dir_enc = validate_format_dir_encoding(dir_enc)
    nabrs, codes, _ = get_neighbour_values(
        dem, dir_enc, include_self=True, pad_val=np.max(dem) + 1
    )
    flow2self_code = np.where(np.all(dir_enc.offsets == [0, 0], axis=1))[0][0]
    dirs = np.full(dem.shape, flow2self_code, dtype=NpFlowDir)
    # find where not all neighbours are nan
    valid_mask = ~np.all(np.isnan(nabrs), axis=0)
    dirs[valid_mask] = np.nanargmin(nabrs[:, valid_mask], axis=0)

    dirs = codes[dirs].astype(NpFlowDir)
    is_flat = dirs == 0
    return dirs, is_flat


def count_indegree(
    dirs: NDArray[NpFlowDir],
    dir_enc: DirectionEncoding | None = None,
    valids: NDArray[np.bool_] | None = None,
) -> NDArray[np.int8]:
    dir_enc = validate_format_dir_encoding(dir_enc)
    valids = validate_format_valids(valids, dirs, "flow direction raster")
    indegs = np.zeros(dirs.shape, dtype=np.int8)
    dsi, dsj, _, ds_valids = compute_downstream_indices(
        dirs, dir_enc=dir_enc, valids=valids, check=False, return_flat_index=False
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
    dirs: NDArray[NpFlowDir],
    indegs: NDArray[np.integer],
    valids: NDArray[np.bool_],
    dir_enc: DirectionEncoding | None = None,
) -> NDArray[np.bool_]:
    """Finds valid cells that do not belong to a directed flow cycle."""
    dir_enc = validate_format_dir_encoding(dir_enc)

    remaining_indegs = np.asarray(indegs, dtype=np.int8).copy()
    acyclics = np.zeros(valids.shape, dtype=bool)
    queue = deque(map(tuple, np.argwhere(valids & (remaining_indegs == 0))))

    dsi, dsj, _, ds_inbounds = compute_downstream_indices(
        dirs,
        dir_enc=dir_enc,
        check=False,
        return_flat_index=False,
        oob_is_okay=True,
    )
    ds_valids = np.zeros(valids.shape, dtype=bool)
    ds_valids[ds_inbounds] = valids[dsi[ds_inbounds], dsj[ds_inbounds]]
    di, dj = dir_enc.code_to_offset(dirs)
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
