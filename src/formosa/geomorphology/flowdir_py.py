# Last modified
#   2026-06-11, En-Chi Lee (williameclee@gmail.com)
#     - Moved Python backend implementations to this file
#     - Removed redundant NaN checks against integer arrays
#     - Standardised variable, argument, and function names
#   2026-07-02, En-Chi Lee (williameclee@gmail.com)
#     - Updated indegree algorithm
#     - Added `_compute_flow_strahler_order_py` and `_construct_flowgraph_py`
#   2026-07-08, En-Chi Lee (williameclee@gmail.com)
#     - Renamed helper submodule from `aux` to `utils`
#   2026-07-09, En-Chi Lee (williameclee@gmail.com)
#     - Added better validity check in `_count_indegree_py`
#   2026-07-12, En-Chi Lee (williameclee@gmail.com)
#     - Implemented Python backend of function `locate_invalid_graph_topology`

import numpy as np

from formosa.geomorphology.d8directions import D8Directions
from .utils import get_neighbour_values, compute_downstream_indices

from formosa.geomorphology.flowdir_f import distances as dist_f

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
    valids: Optional[npt.NDArray[np.integer]] = None,
) -> npt.NDArray[np.int8]:
    if valids is None:
        valids = np.ones(dirs.shape, dtype=bool)
    indegs = np.zeros(dirs.shape, dtype=np.int8)
    dsi, dsj, _, ds_valids = compute_downstream_indices(
        dirs, dir_scheme=dir_scheme, valids=valids, check=False
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
        _, _, dsij, _ = compute_downstream_indices(dirs, dir_scheme=dir_scheme)
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
    downstream_i, downstreamj, _, _ = compute_downstream_indices(
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


def _construct_flowgraph_py(
    dirs: npt.NDArray[np.integer],
    dir_scheme: D8Directions,
    valids: npt.NDArray[np.bool_],
    orders: npt.NDArray[np.integer],
    indegs: npt.NDArray[np.integer],
    seeds: npt.NDArray[np.bool_],
    preserve_junctions: bool = True,
    ncells: Optional[int] = None,
):
    seens = np.zeros_like(dirs, dtype=np.bool_)

    # Hold the cell ijs of the start and end node
    if ncells is None:
        ncells = dirs.size
    arc_orders = np.zeros((ncells,), dtype=np.int8)
    vertex_ijs = np.empty((2, 2 * ncells), dtype=np.int32)
    vertex_startends = np.empty((2, ncells), dtype=np.int32)

    # Find seed cells to start with
    seed_ijs = np.zeros((2, np.sum(valids)), dtype=np.int32, order="F")
    nseeds: int = np.sum(seeds)
    seed_i, seed_j = np.nonzero(seeds)
    seed_ijs[0, :nseeds] = seed_i
    seed_ijs[1, :nseeds] = seed_j

    iseed: int = 0
    iarc: int = 0
    ivertex: int = 0

    while iseed < nseeds:
        si, sj = seed_ijs[0, iseed], seed_ijs[1, iseed]
        iseed += 1
        seens[si, sj] = True

        # Skip isolated point
        di, dj = dir_scheme.code2d8offset(dirs[si, sj])
        if (di == 0) and (dj == 0):
            continue

        # Initialise the arc
        order = orders[si, sj]
        arc_orders[iarc] = order
        vertex_startends[0, iarc] = ivertex
        vertex_ijs[:, ivertex] = [si, sj]
        ivertex += 1
        ci, cj = si, sj

        while True:
            di, dj = dir_scheme.code2d8offset(dirs[ci, cj])
            ni = ci + di
            nj = cj + dj

            ds_is_valid = True
            if (ci == ni) and (cj == nj):  # Self-loop
                ds_is_valid = False
            elif (
                (ni < 0) or (ni >= dirs.shape[0]) or (nj < 0) or (nj >= dirs.shape[1])
            ):  # OOB
                ds_is_valid = False
            elif not valids[ni, nj]:
                ds_is_valid = False

            is_end_vertex = (not ds_is_valid) or (orders[ni, nj] != order)
            if preserve_junctions:
                is_end_vertex = is_end_vertex or (indegs[ni, nj] >= 2)

            if is_end_vertex:
                if not ds_is_valid:
                    if vertex_startends[0, iarc] == ivertex - 1:
                        # Single-length arc, roll back arc and vertex registration
                        ivertex -= 1
                        iarc -= 1
                        break
                    else:
                        vertex_startends[1, iarc] = ivertex - 1
                        break
                vertex_ijs[:, ivertex] = [ni, nj]
                vertex_startends[1, iarc] = ivertex
                ivertex += 1
                if (ds_is_valid) and (not seens[ni, nj]):
                    seens[ni, nj] = True
                    seed_ijs[:, nseeds] = [ni, nj]
                    nseeds += 1
                break

            seens[ni, nj] = True

            vertex_ijs[:, ivertex] = [ni, nj]
            ivertex += 1
            ci, cj = ni, nj
        iarc += 1

    narcs = iarc
    nvertices = ivertex

    return narcs, nvertices, arc_orders, vertex_ijs, vertex_startends


def _locate_invalid_graph_topology_py(
    arc_endpts: npt.NDArray[np.integer], vertex_ijs: npt.NDArray[np.number]
) -> list[tuple[int, int, int, int, int]]:
    narcs = arc_endpts.shape[0]

    # Construct bounding box for each arc: [min_x, min_y, max_x, max_y]
    arc_bboxes = np.empty((narcs, 4), dtype=np.float64)
    for iarc in range(narcs):
        start_idx = arc_endpts[iarc, 0]
        end_idx = arc_endpts[iarc, 1]
        ijs = vertex_ijs[start_idx : end_idx + 1]
        arc_bboxes[iarc, 0] = np.min(ijs[:, 0])
        arc_bboxes[iarc, 1] = np.min(ijs[:, 1])
        arc_bboxes[iarc, 2] = np.max(ijs[:, 0])
        arc_bboxes[iarc, 3] = np.max(ijs[:, 1])

    violations = []

    # Check self-intersections within each arc
    for iarc in range(narcs):
        start_idx = arc_endpts[iarc, 0]
        end_idx = arc_endpts[iarc, 1]
        if end_idx - start_idx <= 1:
            continue
        for iseg in range(start_idx, end_idx):
            for jseg in range(iseg + 1, end_idx):
                intx_flag = dist_f.lines_intersect_v2(
                    vertex_ijs[iseg],
                    vertex_ijs[iseg + 1],
                    vertex_ijs[jseg],
                    vertex_ijs[jseg + 1],
                )
                if intx_flag > 0:
                    violations.append((iarc, iarc, iseg, jseg, intx_flag))

    # Check intersections between different arcs using sweep-line sort and early termination
    idx = np.argsort(arc_bboxes[:, 0])
    for i in range(narcs):
        iarc = idx[i]
        for j in range(i + 1, narcs):
            jarc = idx[j]

            # Since sorted by min x, if min x of right arc is greater than max x of left arc,
            # no subsequent arcs can overlap in x with iarc.
            if arc_bboxes[jarc, 0] > arc_bboxes[iarc, 2]:
                break

            # Check overlap of bounding boxes:
            # if left x > right x or right x < left x or bottom y > top y or top y < bottom y
            if (
                arc_bboxes[iarc, 0] > arc_bboxes[jarc, 2]
                or arc_bboxes[iarc, 2] < arc_bboxes[jarc, 0]
                or arc_bboxes[iarc, 1] > arc_bboxes[jarc, 3]
                or arc_bboxes[iarc, 3] < arc_bboxes[jarc, 1]
            ):
                continue

            start_i = arc_endpts[iarc, 0]
            end_i = arc_endpts[iarc, 1]
            start_j = arc_endpts[jarc, 0]
            end_j = arc_endpts[jarc, 1]

            for iseg in range(start_i, end_i):
                for jseg in range(start_j, end_j):
                    intx_flag = dist_f.lines_intersect_v2(
                        vertex_ijs[iseg],
                        vertex_ijs[iseg + 1],
                        vertex_ijs[jseg],
                        vertex_ijs[jseg + 1],
                    )
                    if intx_flag > 0:
                        if iarc < jarc:
                            violations.append((iarc, jarc, iseg, jseg, intx_flag))
                        else:
                            violations.append((jarc, iarc, jseg, iseg, intx_flag))

    return violations
