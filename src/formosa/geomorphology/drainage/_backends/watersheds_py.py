# Last modified
#   2026-06-11, En-Chi Lee (williameclee@gmail.com)
#     - Moved Python backend implementations to this file
#     - Removed redundant NaN checks against integer arrays
#     - Standardised variable, argument, and function names
#   2026-07-30, En-Chi Lee (williameclee@gmail.com)
#     - Fixed Python/FORTRAN backend behaviour parity in
#       `compute_flow_strahler_order`.

import numpy as np

from formosa.geomorphology.drainage.directions import D8Directions
from formosa.geomorphology.drainage.utils import compute_downstream_indices
import formosa.geomorphology.drainage._backends.flowdir_py as flowdir_py

from typing import Optional
import numpy.typing as npt


def compute_flow_accumulation(
    dirs: npt.NDArray[np.integer],
    valids: Optional[npt.NDArray[np.bool_]] = None,
    weights: Optional[npt.NDArray[np.floating]] = None,
    indegs: Optional[npt.NDArray[np.integer]] = None,
    dsij: Optional[npt.NDArray[np.integer]] = None,
    dir_scheme: D8Directions = D8Directions(),
) -> np.ndarray:
    from collections import deque

    # Initialisation
    I, J = dirs.shape

    if indegs is None:
        indegs = flowdir_py.count_indegree(dirs, dir_scheme=dir_scheme, valids=valids)
    else:
        assert (
            indegs.shape == dirs.shape
        ), f"Shape for flowdirs and indegree must match, but got indegree shape {indegs.shape} and flowdirs shape {dirs.shape} instead"

    if valids is None:
        valids = (dirs != 0) | (indegs > 0)
    else:
        assert (
            valids.shape == dirs.shape
        ), f"Shape for flowidr and valid mask must match, but got valid shape {valids.shape} and flowdirs shape {dirs.shape} instead"
    if weights is None:
        weights = np.where(valids, 1, 0).astype(np.uint64)  # type: ignore
    else:
        assert (
            weights.shape == dirs.shape
        ), f"Shape for flowdirs and weight must match, but got weight shape {weights.shape} and flowdirs shape {dirs.shape} instead"
        weights = np.where(valids, weights, 0)  # type: ignore

    if dsij is None:
        _, _, dsij, _ = compute_downstream_indices(
            dirs, dir_scheme=dir_scheme, return_flat_index=True
        )
    else:
        assert (
            dsij.shape == dirs.shape
        ), f"Shape for flowdirs and downstream ij indices must match, but got dsij: {dsij.shape} and flowdirs: {dirs.shape} instead"

    indegs = indegs.flatten(order="F")
    valids = valids.flatten(order="F")  # type: ignore
    weights = weights.flatten(order="F")  # type: ignore
    dsij = dsij.flatten(order="F")  # type: ignore ; dsij will not be None
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
    dirs: npt.NDArray[np.integer],
    dir_scheme: D8Directions = D8Directions(),
    valids: Optional[npt.NDArray[np.bool_]] = None,
    indegs: Optional[npt.NDArray[np.integer]] = None,
) -> npt.NDArray[np.int16]:
    from collections import deque

    if valids is None:
        valids = np.ones(dirs.shape, dtype=bool)
    if indegs is None:
        indegs = flowdir_py.count_indegree(dirs, dir_scheme=dir_scheme, valids=valids)
    else:
        indegs = indegs.copy()

    downstream_i, downstream_j, _, downstream_valids = compute_downstream_indices(
        dirs, dir_scheme=dir_scheme, valids=valids, check=False, return_flat_index=False
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


def label_watersheds(
    dirs: npt.NDArray[np.integer],
    dir_scheme: D8Directions = D8Directions(),
    valids: Optional[npt.NDArray[np.bool_]] = None,
) -> npt.NDArray[np.int32]:
    if valids is None:
        valids = ~np.isnan(dirs)
    elif isinstance(valids, np.ndarray):
        assert (
            valids.shape == dirs.shape
        ), f"Shape for flow direction ({valids.shape}) and valid mask ({dirs.shape}) do not match."
        # Removed the check for NaN values in flowdirs, since integer types cannot hold NaN anyway
    else:
        raise TypeError(
            f"[FORMOSA] VALIDS must be either None or a numpy array, got {type(valids)} instead."
        )

    I, J = dirs.shape
    ii, jj = np.meshgrid(
        np.arange(I, dtype=np.int32), np.arange(J, dtype=np.int32), indexing="ij"
    )
    codes: list[int] = dir_scheme.codes.tolist()
    offsets: list[tuple[int, int]] = [
        (int(di), int(dj)) for di, dj in dir_scheme.offsets.astype(np.int32, copy=False)
    ]

    seeds: list[tuple[int, int]] = list(
        zip(ii[valids & (dirs == 0)], jj[valids & (dirs == 0)])
    )

    watershed = -np.ones(dirs.shape, dtype=np.int32)

    for label, seed in enumerate(seeds):
        to_fill: list[tuple[int, int]] = [seed]

        while to_fill:
            ci, cj = to_fill.pop(0)
            watershed[ci, cj] = label
            for code, (di, dj) in zip(codes, offsets):
                ni, nj = ci - di, cj - dj
                if (ni < 0 or ni >= I) or (nj < 0 or nj >= J):
                    continue
                elif not valids[ni, nj]:
                    continue
                elif watershed[ni, nj] != -1:
                    continue

                if dirs[ni, nj] == code:
                    to_fill.append((ni, nj))
    watershed = watershed + 1  # make background 0 and watersheds start from 1
    return watershed
