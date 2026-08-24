"""
Computes flow-based raster metrics using the Python backend.

This module implements internal routines called by the public-facing
drainage API and is not intended to be used directly.

Last modified: 2026-08-24, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np
from numpy.typing import NDArray

from formosa.geomorphology.drainage.directions import D8Directions
from formosa.geomorphology.drainage.neighbours import (
    compute_downstream_indices,
)
from formosa.utils import NpFlowDir


def compute_flow_accumulation(
    dirs: NDArray[NpFlowDir],
    valids: NDArray[np.bool_],
    weights: NDArray[np.floating],
    indegs: NDArray[np.integer],
    dsij: NDArray[np.integer],
    dir_scheme: D8Directions | None = None,
) -> np.ndarray:
    from collections import deque

    # Initialisation
    I, J = dirs.shape

    indegs = indegs.flatten(order="F")
    valids = valids.flatten(order="F")
    weights = weights.flatten(order="F")
    dsij = dsij.flatten(order="F")
    dirs = dirs.flatten(order="F")

    # Initialize accumulation with self weight
    accumulation = weights.ravel().astype(weights.dtype, copy=True)

    # Queue sources (indeg == 0) among valid cells
    q = deque(np.flatnonzero((indegs == 0) & valids))

    # Topological propagation
    while q:
        u = q.popleft()
        v = dsij[u]
        if not valids[v]:
            continue
        accumulation[v] += accumulation[u]
        indegs[v] -= 1
        if indegs[v] == 0:
            q.append(v)

    accumulation = accumulation.reshape(I, J, order="F")

    return accumulation


def compute_flow_strahler_order(
    dirs: NDArray[NpFlowDir],
    dir_scheme: D8Directions,
    valids: NDArray[np.bool_],
    indegs: NDArray[np.integer],
) -> NDArray[np.int16]:
    from collections import deque

    indegs = indegs.copy()

    downstream_i, downstream_j, _, downstream_valids = (
        compute_downstream_indices(
            dirs,
            dir_scheme,
            valids=valids,
            check=False,
            return_flat_index=False,
        )
    )

    strahler_order = np.zeros(indegs.shape, dtype=np.int16)
    seeds_mask = valids & (indegs == 0)
    strahler_order[seeds_mask] = 1

    max_upstream_order = np.zeros(indegs.shape, dtype=np.int16)
    max_upstream_count = np.zeros(indegs.shape, dtype=np.int8)

    ii, jj = np.indices(indegs.shape, dtype=np.int32)
    seeds = deque(zip(ii[seeds_mask], jj[seeds_mask]))  # type: ignore

    while seeds:
        ci, cj = seeds.popleft()
        dsi, dsj = downstream_i[ci, cj], downstream_j[ci, cj]
        if (
            not downstream_valids[ci, cj]
            or not valids[dsi, dsj]
            or (ci, cj) == (dsi, dsj)
        ):
            continue

        upstream_order = strahler_order[ci, cj]
        if upstream_order > max_upstream_order[dsi, dsj]:
            max_upstream_order[dsi, dsj] = upstream_order
            max_upstream_count[dsi, dsj] = 1
        elif upstream_order == max_upstream_order[dsi, dsj]:
            max_upstream_count[dsi, dsj] += 1

        indegs[dsi, dsj] -= 1
        if indegs[dsi, dsj] == 0:
            strahler_order[dsi, dsj] = max_upstream_order[dsi, dsj]
            if max_upstream_count[dsi, dsj] >= 2:
                strahler_order[dsi, dsj] += 1
            seeds.append((dsi, dsj))

    return strahler_order
