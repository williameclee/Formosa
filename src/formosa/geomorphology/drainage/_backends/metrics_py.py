"""
Computes flow-based raster metrics using the Python backend.

This module implements internal routines called by the public-facing
drainage API and is not intended to be used directly.

Last modified: 2026-08-24, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np
from numpy.typing import NDArray

from formosa.geomorphology.drainage.directions import DirectionEncoding
from formosa.geomorphology.drainage.neighbours import compute_downstream_indices
from formosa.utils import NpFlowDir


def compute_flow_accumulation[W: np.floating](
    dirs: NDArray[NpFlowDir],
    valids: NDArray[np.bool_],
    wgts: NDArray[W],
    indegs: NDArray[np.integer],
    dsij: NDArray[np.integer],
) -> NDArray[W]:
    from collections import deque

    # Initialisation
    I, J = dirs.shape

    indegs = indegs.flatten(order="F")
    valids = valids.flatten(order="F")
    wgts = wgts.flatten(order="F")
    dsij = dsij.flatten(order="F")
    dirs = dirs.flatten(order="F")

    # Initialise accumulation with self weight
    accums = wgts.ravel().astype(wgts.dtype, copy=True)

    # Queue sources (indeg == 0) among valid cells
    q = deque(np.flatnonzero((indegs == 0) & valids))

    # Topological propagation
    while q:
        u = q.popleft()
        v = dsij[u]
        if not valids[v]:
            continue
        accums[v] += accums[u]
        indegs[v] -= 1
        if indegs[v] == 0:
            q.append(v)

    accums = accums.reshape(I, J, order="F")

    return accums


def compute_flow_strahler_order(
    dirs: NDArray[NpFlowDir],
    dir_enc: DirectionEncoding,
    valids: NDArray[np.bool_],
    indegs: NDArray[np.integer],
) -> NDArray[np.int16]:
    from collections import deque

    indegs = indegs.copy()

    dsis, dsjs, _, ds_valids = compute_downstream_indices(
        dirs, dir_enc, valids=valids, check=False, return_flat_index=False
    )

    orders = np.zeros(indegs.shape, dtype=np.int16)
    seeds_mask = valids & (indegs == 0)
    orders[seeds_mask] = 1

    max_upstrm_order = np.zeros(indegs.shape, dtype=np.int16)
    max_upstrm_cnt = np.zeros(indegs.shape, dtype=np.int8)

    ii, jj = np.indices(indegs.shape, dtype=np.int32)
    seeds = deque(zip(ii[seeds_mask], jj[seeds_mask]))  # type: ignore

    while seeds:
        ci, cj = seeds.popleft()
        dsi = dsis[ci, cj]
        dsj = dsjs[ci, cj]
        if not ds_valids[ci, cj] or not valids[dsi, dsj] or (ci, cj) == (dsi, dsj):
            continue

        upstrm_order = orders[ci, cj]
        if upstrm_order > max_upstrm_order[dsi, dsj]:
            max_upstrm_order[dsi, dsj] = upstrm_order
            max_upstrm_cnt[dsi, dsj] = 1
        elif upstrm_order == max_upstrm_order[dsi, dsj]:
            max_upstrm_cnt[dsi, dsj] += 1

        indegs[dsi, dsj] -= 1
        if indegs[dsi, dsj] == 0:
            orders[dsi, dsj] = max_upstrm_order[dsi, dsj]
            if max_upstrm_cnt[dsi, dsj] >= 2:
                orders[dsi, dsj] += 1
            seeds.append((dsi, dsj))

    return orders
