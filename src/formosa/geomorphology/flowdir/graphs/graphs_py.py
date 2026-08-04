# Last modified
#   2026-07-12, En-Chi Lee (williameclee@gmail.com)
#     - Implemented Python backend of function `locate_invalid_graph_topology`
#   2026-07-14, En-Chi Lee (williameclee@gmail.com)
#     - Splitted `geomorphology.flowdir` into submodules
#   2026-07-30, En-Chi Lee (williameclee@gmail.com)
#     - Various minor refactors and type annotation enhancements

import numpy as np

from formosa.geomorphology.flowdir.d8directions import D8Directions
from formosa.geomorphology.distances_py import _lines_intersect_v2


import numpy.typing as npt
from typing import Optional


def _construct_flowgraph_py(
    dirs: npt.NDArray[np.integer],
    dir_scheme: D8Directions,
    valids: npt.NDArray[np.bool_],
    orders: npt.NDArray[np.integer],
    indegs: npt.NDArray[np.integer],
    seeds: npt.NDArray[np.bool_],
    preserve_junctions: bool = True,
    ncells: Optional[int] = None,
) -> tuple[
    int, int, npt.NDArray[np.int8], npt.NDArray[np.int32], npt.NDArray[np.int32]
]:
    seens = np.zeros_like(dirs, dtype=np.bool_)

    # Hold the cell ijs of the start and end node
    if ncells is None:
        ncells = dirs.size
    graph_orders = np.zeros((ncells,), dtype=np.int8)
    graph_verts = np.empty((2, 2 * ncells), dtype=np.int32)
    graph_endpts = np.empty((2, ncells), dtype=np.int32)

    # Find seed cells to start with
    seed_ijs = np.zeros((2, int(np.sum(valids))), dtype=np.int32, order="F")
    nseeds = int(np.sum(seeds))
    seed_i, seed_j = np.nonzero(seeds)
    seed_ijs[0, :nseeds] = seed_i
    seed_ijs[1, :nseeds] = seed_j

    iseed: int = 0
    iarc: int = 0
    ivert: int = 0

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
        graph_orders[iarc] = order
        graph_endpts[0, iarc] = ivert
        graph_verts[:, ivert] = [si, sj]
        ivert += 1
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
                    if graph_endpts[0, iarc] == ivert - 1:
                        # Single-length arc, roll back arc and vertex registration
                        ivert -= 1
                        iarc -= 1
                        break
                    else:
                        graph_endpts[1, iarc] = ivert - 1
                        break
                graph_verts[:, ivert] = [ni, nj]
                graph_endpts[1, iarc] = ivert
                ivert += 1
                if (ds_is_valid) and (not seens[ni, nj]):
                    seens[ni, nj] = True
                    seed_ijs[:, nseeds] = [ni, nj]
                    nseeds += 1
                break

            seens[ni, nj] = True

            graph_verts[:, ivert] = [ni, nj]
            ivert += 1
            ci, cj = ni, nj
        iarc += 1

    narcs = iarc
    nverts = ivert

    return narcs, nverts, graph_orders, graph_verts, graph_endpts


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
                intx_flag = _lines_intersect_v2(
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
                    intx_flag = _lines_intersect_v2(
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
