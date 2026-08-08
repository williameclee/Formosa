# Last modified
#   2026-06-11, En-Chi Lee (williameclee@gmail.com)
#     - Moved Python backend implementations to this file
#     - Removed redundant NaN checks against integer arrays
#     - Standardised variable, argument, and function names
#   2026-07-02, En-Chi Lee (williameclee@gmail.com)
#     - Updated indegree algorithm
#   2026-07-08, En-Chi Lee (williameclee@gmail.com)
#     - Renamed helper submodule from `aux` to `utils`
#   2026-07-09, En-Chi Lee (williameclee@gmail.com)
#     - Added better validity check in `_count_indegree_py`
#   2026-07-14, En-Chi Lee (williameclee@gmail.com)
#     - Splitted `geomorphology.flowdir` into submodules
#   2026-07-30, En-Chi Lee (williameclee@gmail.com)
#     - Fixed Python/FORTRAN backend behaviour parity in 
#       `compute_flow_strahler_order`.
#   2026-08-03, En-Chi Lee (williameclee@gmail.com)
#     - Implemented Python backend for function 
#       `find_acyclic_flowdirs`.

from collections import deque

import numpy as np

from formosa.geomorphology.flowdir.utils import (
    get_neighbour_values,
    compute_downstream_indices,
)
from formosa.geomorphology.flowdir.directions import D8Directions

import numpy.typing as npt
from typing import Optional


def _compute_flowdir_simple_py(
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


def _compute_masked_flowdir_py(
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


def _count_indegree_py(
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


def _find_acyclic_flowdirs_py(
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


def _find_flat_edges_py(
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


def _compute_flow_accumulation_py(
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
        indegs = _count_indegree_py(dirs, dir_scheme=dir_scheme)
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


def _compute_flow_strahler_order_py(
    dirs: npt.NDArray[np.integer],
    dir_scheme: D8Directions = D8Directions(),
    valids: Optional[npt.NDArray[np.bool_]] = None,
    indegs: Optional[npt.NDArray[np.integer]] = None,
) -> npt.NDArray[np.int16]:
    from collections import deque

    if valids is None:
        valids = np.ones(dirs.shape, dtype=bool)
    if indegs is None:
        indegs = _count_indegree_py(dirs, dir_scheme=dir_scheme, valids=valids)
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


def _label_watersheds_py(
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
