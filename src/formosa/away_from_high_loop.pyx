# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np

ctypedef np.int32_t INT32_t


def _away_From_high_loop(
    INT32_t[:, :] flowdirs,          # 2D int32
    INT32_t[:, :] flat_mask,         # 2D int32
    INT32_t[:] flat_height,          # 1D int32, indexed by labels
    INT32_t[:, :] labels,            # 2D int32
    list high_edges,                 # list[tuple[int, int]]
    INT32_t[:, :] offsets,           # (n_offsets, 2)
    tuple shape_ij,                  # (n_i, n_j)
):
    cdef:
        Py_ssize_t shape_i = shape_ij[0]
        Py_ssize_t shape_j = shape_ij[1]
        Py_ssize_t ci, cj, ni, nj
        Py_ssize_t di, dj
        Py_ssize_t k, n_offsets = offsets.shape[0]
        Py_ssize_t max_len = shape_i * shape_j * 2

        INT32_t loops = 1
        INT32_t label_idx

        object cij                  # holds the popped tuple
        tuple marker = (-1, -1)

    # Append sentinel marker
    high_edges.append(marker)

    while len(high_edges) > 1:
        # Python list used as a queue
        cij = high_edges.pop(0)
        ci = <Py_ssize_t>cij[0]
        cj = <Py_ssize_t>cij[1]

        # Check for sentinel
        if ci == -1 and cj == -1:
            if len(high_edges) == 1:
                break
            loops += 1
            high_edges.append(marker)
            continue

        if flat_mask[ci, cj] != 0:
            continue

        flat_mask[ci, cj] = loops

        label_idx = labels[ci, cj]
        flat_height[label_idx] = loops

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

            high_edges.append((ni, nj))

        if len(high_edges) > max_len:
            raise RuntimeError(
                "Possible infinite loop detected in _away_From_high_loop"
            )

    # memoryviews wrap the original NumPy arrays, so convert back to ndarrays
    return np.asarray(flat_mask), np.asarray(flat_height)