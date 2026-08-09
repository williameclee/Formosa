"""
Conversions of a flow direction raster to a flow graph.

Content of this file is mostly designed to be called by the public-
facing APIs and not directly by the user.

Last modified: 2026-07-30, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np

from formosa.geomorphology.drainage.directions import D8Directions


import numpy.typing as npt
from typing import Optional


def construct_flowgraph(
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
