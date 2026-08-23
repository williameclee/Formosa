"""
Constructs flow graphs from direction rasters using the Python
backend.

This module implements internal routines called by the public-facing
network API and is not intended to be used directly.

Last modified: 2026-08-23, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np
from numpy.typing import NDArray

from formosa.geomorphology.drainage.directions import D8Directions
from formosa.utils import NpFlowDir


def construct_flowgraph(
    dirs: NDArray[NpFlowDir],
    dir_scheme: D8Directions,
    valids: NDArray[np.bool_],
    orders: NDArray[np.integer],
    indegs: NDArray[np.integer],
    seeds: NDArray[np.bool_],
    preserve_junctions: bool = True,
    ncells: int | None = None,
) -> tuple[int, int, NDArray[np.int8], NDArray[np.int32], NDArray[np.int32]]:
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
