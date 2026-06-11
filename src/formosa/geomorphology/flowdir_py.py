# Last modified
#   2026-06-11, En-Chi Lee (williameclee@gmail.com)
#     - Moved Python backend implementations to this file

import numpy as np

from formosa.geomorphology.d8directions import D8Directions
import numpy.typing as npt
from typing import Optional


def _compute_downstream_indices_py(
    flowdirs: npt.NDArray[np.integer],
    directions: D8Directions = D8Directions(),
    valids: Optional[npt.NDArray[np.bool_]] = None,
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer], npt.NDArray[np.int32]]:
    """
    Computes the downstream indices for each cell in a flow direction grid.

    Parameters
    ----------
    flowdirs : NDArray[int]
        A 2D array representing the flow directions for each cell.
    directions : D8Directions, optional
        An instance of D8Directions defining the flow direction scheme.
        Default is D8Directions().
    valids : NDArray[bool], optional
        A boolean mask array indicating valid cells in the flow direction grid.
        If None, all cells are considered valid.
        Default is None.

    Returns
    -------
    dsi : NDArray[int]
        A 2D array of downstream row indices for each cell.
    dsj : NDArray[int]
        A 2D array of downstream column indices for each cell.
    dsij : NDArray[int32]
        A 2D array of flattened downstream indices for each cell.
    """
    if valids is None:
        valids = ~np.isnan(flowdirs)
    elif isinstance(valids, np.ndarray):
        assert (
            valids.shape == flowdirs.shape
        ), f"Shapes for flow direction ({flowdirs.shape}) and valid mask ({valids.shape}) do not match."
    else:
        raise TypeError(
            f"Expected valids to be None or np.ndarray, got {type(valids)} instead."
        )

    I, J = flowdirs.shape
    ii, jj = np.meshgrid(
        np.arange(I, dtype=np.int32), np.arange(J, dtype=np.int32), indexing="ij"
    )
    di, dj = directions.code2d8offset(flowdirs)
    dsi = (ii.astype(np.int16) + (di).astype(np.int16)).astype(np.int16)
    dsj = (jj.astype(np.int16) + (dj).astype(np.int16)).astype(np.int16)
    dsij: npt.NDArray[np.int32] = dsj.astype(np.int32) * I + dsi.astype(np.int32)

    if np.any((dsi < 0) | (dsi >= I) | (dsj < 0) | (dsj >= J)):
        raise ValueError("Some downstream indices out of bounds")

    return dsi, dsj, dsij


def _compute_indegree_py(
    flowdirs: npt.NDArray[np.integer], directions: D8Directions = D8Directions()
) -> npt.NDArray[np.integer]:
    indegree = np.zeros(flowdirs.shape, dtype=np.int32)
    dsi, dsj, _ = _compute_downstream_indices_py(flowdirs, directions=directions)

    for flowdir in np.unique(flowdirs):
        if flowdir == 0:
            continue
        is_Valid_ds = (
            (flowdirs == flowdir)
            & (dsi >= 0)
            & (dsi < flowdirs.shape[0])
            & (dsj >= 0)
            & (dsj < flowdirs.shape[1])
        )
        indegree[dsi[is_Valid_ds], dsj[is_Valid_ds]] += 1

    return indegree


def _compute_flow_accumulation_py(
    flowdirs: npt.NDArray[np.integer],
    valids: Optional[npt.NDArray[np.bool_]] = None,
    weights: Optional[npt.NDArray[np.floating]] = None,
    indegrees: Optional[npt.NDArray[np.integer]] = None,
    dsij: Optional[npt.NDArray[np.integer]] = None,
    directions: D8Directions = D8Directions(),
) -> np.ndarray:
    from collections import deque

    # Initialisation
    I, J = flowdirs.shape

    if indegrees is None:
        indegrees = _compute_indegree_py(flowdirs, directions=directions)
    else:
        assert (
            indegrees.shape == flowdirs.shape
        ), f"Shape for flowdir and indegree must match, but got indegree shape {indegrees.shape} and flowdir shape {flowdirs.shape} instead"

    if valids is None:
        valids = (flowdirs != 0) | (indegrees > 0)
    else:
        assert (
            valids.shape == flowdirs.shape
        ), f"Shape for flowidr and valid mask must match, but got valid shape {valids.shape} and flowdir shape {flowdirs.shape} instead"
    if weights is None:
        weights = np.where(valids, 1, 0).astype(np.uint64)  # type: ignore
    else:
        assert (
            weights.shape == flowdirs.shape
        ), f"Shape for flowdir and weight must match, but got weight shape {weights.shape} and flowdir shape {flowdirs.shape} instead"
        weights = np.where(valids, weights, 0)  # type: ignore

    if dsij is None:
        _, _, dsij = _compute_downstream_indices_py(flowdirs, directions=directions)
    else:
        assert (
            dsij.shape == flowdirs.shape
        ), f"Shape for flowdir and downstream ij indices must match, but got dsij: {dsij.shape} and flowdir: {flowdirs.shape} instead"

    indegrees = indegrees.flatten(order="F")
    valids = valids.flatten(order="F")  # type: ignore
    weights = weights.flatten(order="F")  # type: ignore
    dsij = dsij.flatten(order="F")
    flowdirs = flowdirs.flatten(order="F")

    # Initialize accumulation with self weight
    accumulation = weights.ravel().astype(weights.dtype, copy=True)

    # Queue sources (indeg == 0) among valid cells
    q = deque(np.flatnonzero((indegrees == 0) & valids))

    # Topological propagation
    while q:
        u = q.popleft()
        v = dsij[u]
        if not valids[v]:
            continue
        accumulation[v] += accumulation[u]
        indegrees[v] -= 1
        if indegrees[v] == 0:
            q.append(v)

    accumulation = accumulation.reshape(I, J, order="F")

    return accumulation


def _compute_strahler_order_py(
    flowdir: npt.NDArray[np.integer],
    directions: D8Directions = D8Directions(),
    indegrees: Optional[npt.NDArray[np.integer]] = None,
) -> npt.NDArray[np.int16]:
    from collections import deque

    if indegrees is None:
        from .flowdir_py import _compute_indegree_py

        indegrees = _compute_indegree_py(flowdir, directions=directions)
    downstream_i, downstreamj, _ = _compute_downstream_indices_py(
        flowdir, directions=directions
    )

    strahler_order = np.zeros(indegrees.shape, dtype=np.int16)
    strahler_order[indegrees == 0] = 1

    ii, jj = np.indices(indegrees.shape, dtype=np.int32)
    seeds = deque(zip(ii[indegrees == 0], jj[indegrees == 0]))  # type: ignore TODO: figure out what the type error actually is

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
        indegrees[dsi, dsj] -= 1
        if indegrees[dsi, dsj] == 0:
            seeds.append((dsi, dsj))
    return strahler_order


def _label_watersheds_py(
    flowdir: npt.NDArray[np.integer],
    directions: D8Directions = D8Directions(),
    valids: Optional[npt.NDArray[np.bool_]] = None,
) -> npt.NDArray[np.int32]:
    if valids is None:
        valids = ~np.isnan(flowdir)
    elif isinstance(valids, np.ndarray):
        assert (
            valids.shape == flowdir.shape
        ), f"Shape for flow direction ({valids.shape}) and valid mask ({flowdir.shape}) do not match."
        valids = valids.astype(bool, copy=False) & (~np.isnan(flowdir))
        flowdir = np.where(valids, flowdir, np.nan)
    else:
        raise TypeError(
            f"[FORMOSA] VALIDS must be either None or a numpy array, got {type(valids)} instead."
        )

    I, J = flowdir.shape
    ii, jj = np.meshgrid(
        np.arange(I, dtype=np.int32), np.arange(J, dtype=np.int32), indexing="ij"
    )
    codes: list[int] = directions.codes.tolist()
    offsets: list[tuple[int, int]] = [
        (int(di), int(dj)) for di, dj in directions.offsets.astype(np.int32, copy=False)
    ]

    seeds: list[tuple[int, int]] = list(
        zip(ii[valids & (flowdir == 0)], jj[valids & (flowdir == 0)])
    )

    watershed = -np.ones(flowdir.shape, dtype=np.int32)

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

                if flowdir[ni, nj] == code:
                    to_fill.append((ni, nj))
    watershed = watershed + 1  # make background 0 and watersheds start from 1
    return watershed
