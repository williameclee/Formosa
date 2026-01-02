# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np

ctypedef np.int32_t INT32_t

def _flow_distance_loop(
    INT32_t[:, :] distance,         # 2D int32 array (NumPy accepted)
    list seeds,                     # list[tuple[int, int]]
    INT32_t[:, :] offsets,          # shape (n_offsets, 2)
    INT32_t[:, :] downstream_i,     # 2D int32
    INT32_t[:, :] downstream_j,     # 2D int32
    tuple shape,                    # (shape_i, shape_j)
):
    cdef Py_ssize_t ci, cj, ni, nj
    cdef Py_ssize_t di, dj
    cdef Py_ssize_t k
    cdef Py_ssize_t n_offsets = offsets.shape[0]
    cdef Py_ssize_t shape_i = shape[0]
    cdef Py_ssize_t shape_j = shape[1]

    while seeds:
        # seeds is still a Python list of (i, j) tuples
        ci, cj = seeds.pop()

        for k in range(n_offsets):
            di = offsets[k, 0]
            dj = offsets[k, 1]

            ni = ci + di
            nj = cj + dj

            if ni < 0 or ni >= shape_i or nj < 0 or nj >= shape_j:
                continue

            if distance[ni, nj] > 0:
                continue

            if downstream_i[ni, nj] != ci or downstream_j[ni, nj] != cj:
                continue

            distance[ni, nj] = distance[ci, cj] + 1
            seeds.append((ni, nj))

    # distance is a memoryview backed by the original NumPy array,
    # so returning it is fine – Python will see it as a NumPy array.
    return np.asarray(distance)