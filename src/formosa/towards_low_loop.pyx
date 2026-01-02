# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np

ctypedef np.int32_t INT32_t


def _towards_low_loop(
    INT32_t[:, :] flowdirs,          # 2D int32
    INT32_t[:, :] flat_mask,         # 2D int32
    INT32_t[:] flat_height,          # 1D int32
    INT32_t[:, :] labels,            # 2D int32
    list low_edges,                  # list[tuple[int, int]]
    INT32_t[:, :] offsets,           # (n_offsets, 2)
    tuple shape_ij,                  # (ni, nj)
    int step_size,
):
    cdef:
        Py_ssize_t shape_i = shape_ij[0]
        Py_ssize_t shape_j = shape_ij[1]

        Py_ssize_t ci, cj, ni, nj
        Py_ssize_t di, dj
        Py_ssize_t k, n_offsets = offsets.shape[0]

        INT32_t loops = 1
        INT32_t label_idx

        object cij
        tuple marker = (-1, -1)

    # Append sentinel marker
    low_edges.append(marker)

    while len(low_edges) > 1:
        cij = low_edges.pop(0)
        ci = <Py_ssize_t>cij[0]
        cj = <Py_ssize_t>cij[1]

        # Sentinel check
        if ci == -1 and cj == -1:
            loops += 1
            low_edges.append(marker)
            continue

        if flat_mask[ci, cj] > 0:
            continue
        elif flat_mask[ci, cj] < 0:
            label_idx = labels[ci, cj]
            flat_mask[ci, cj] += flat_height[label_idx] + loops * step_size
        else:
            flat_mask[ci, cj] = loops * step_size

        label_idx = labels[ci, cj]

        for k in range(n_offsets):
            di = offsets[k, 0]
            dj = offsets[k, 1]

            ni = ci + di
            nj = cj + dj

            if ni < 0 or ni >= shape_i or nj < 0 or nj >= shape_j:
                continue

            if labels[ni, nj] != label_idx:
                continue

            if flowdirs[ni, nj] != 0:
                continue

            low_edges.append((ni, nj))

    return np.asarray(flat_mask)