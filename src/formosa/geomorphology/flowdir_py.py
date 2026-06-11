# Last modified
#   2026-06-11, En-Chi Lee (williameclee@gmail.com)
#     - Moved Python backend implementations to this file
#     - Removed redundant NaN checks against integer arrays
#     - Standardised variable, argument, and function names

import numpy as np

from formosa.geomorphology.d8directions import D8Directions
from .aux import get_neighbour_values, compute_downstream_indices

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
    dirs: npt.NDArray[np.integer], dir_scheme: D8Directions = D8Directions()
) -> npt.NDArray[np.integer]:
    indegree = np.zeros(dirs.shape, dtype=np.int32)
    dsi, dsj, _ = compute_downstream_indices(dirs, dir_scheme=dir_scheme)

    for flowdir in np.unique(dirs):
        if flowdir == 0:
            continue
        is_Valid_ds = (
            (dirs == flowdir)
            & (dsi >= 0)
            & (dsi < dirs.shape[0])
            & (dsj >= 0)
            & (dsj < dirs.shape[1])
        )
        indegree[dsi[is_Valid_ds], dsj[is_Valid_ds]] += 1

    return indegree


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
        _, _, dsij = compute_downstream_indices(dirs, dir_scheme=dir_scheme)
    else:
        assert (
            dsij.shape == dirs.shape
        ), f"Shape for flowdirs and downstream ij indices must match, but got dsij: {dsij.shape} and flowdirs: {dirs.shape} instead"

    indegs = indegs.flatten(order="F")
    valids = valids.flatten(order="F")  # type: ignore
    weights = weights.flatten(order="F")  # type: ignore
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


def _compute_flow_strahler_order_py(
    dirs: npt.NDArray[np.integer],
    dir_scheme: D8Directions = D8Directions(),
    indegs: Optional[npt.NDArray[np.integer]] = None,
) -> npt.NDArray[np.int16]:
    from collections import deque

    if indegs is None:
        indegs = _count_indegree_py(dirs, dir_scheme=dir_scheme)
    downstream_i, downstreamj, _ = compute_downstream_indices(
        dirs, dir_scheme=dir_scheme
    )

    strahler_order = np.zeros(indegs.shape, dtype=np.int16)
    strahler_order[indegs == 0] = 1

    ii, jj = np.indices(indegs.shape, dtype=np.int32)
    seeds = deque(zip(ii[indegs == 0], jj[indegs == 0]))  # type: ignore TODO: figure out what the type error actually is

    while seeds:
        ci, cj = seeds.popleft()
        dsi, dsj = (
            downstream_i[ci, cj],
            downstreamj[ci, cj],
        )
        if (ci, cj) == (dsi, dsj):
            continue
        if strahler_order[dsi, dsj] < strahler_order[ci, cj]:
            strahler_order[dsi, dsj] = strahler_order[ci, cj]
        else:
            strahler_order[dsi, dsj] += 1
        indegs[dsi, dsj] -= 1
        if indegs[dsi, dsj] == 0:
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
