"""
Validates flow-graph topology using the Python backend.

This module implements internal routines called by the public-facing
network API and is not intended to be used directly.

Last modified: 2026-08-23, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np
from numpy.typing import NDArray

from formosa.geomorphology.geometry.intersections import lines_intersect
from formosa.utils import NpCoords


def locate_invalid_graph_topology(
    endpts: NDArray[np.integer], vtxs: NDArray[NpCoords]
) -> list[tuple[int, int, int, int, int]]:
    """
    Locates self-intersections and cross-arc intersections.

    Parameters
    ----------
    endpts : NDArray[int]
        Start and end vertex index for each arc in `vtxs`.
        - Expected shape: `(narcs, 2)`.
    vtxs : NDArray[float]
        Grid coordinates (i, j) of each vertex.
        - Expected shape: `(nvtxs, 2)`.

    Returns
    -------
    violations : list[tuple[int, int, int, int, int]]
        List of detected violation tuples:
        `(iarc, jarc, iseg, jseg, intx_flag)`.
    """
    narcs = endpts.shape[0]

    # Construct bounding box for each arc: [min_x, min_y, max_x, max_y]
    arc_bboxes = np.empty((narcs, 4), dtype=np.float64)
    for iarc in range(narcs):
        start_idx = endpts[iarc, 0]
        end_idx = endpts[iarc, 1]
        ijs = vtxs[start_idx : end_idx + 1]
        arc_bboxes[iarc, 0] = np.min(ijs[:, 0])
        arc_bboxes[iarc, 1] = np.min(ijs[:, 1])
        arc_bboxes[iarc, 2] = np.max(ijs[:, 0])
        arc_bboxes[iarc, 3] = np.max(ijs[:, 1])

    violations = []

    # Check self-intersections within each arc
    for iarc in range(narcs):
        start_idx = endpts[iarc, 0]
        end_idx = endpts[iarc, 1]
        if end_idx - start_idx <= 1:
            continue
        for iseg in range(start_idx, end_idx):
            for jseg in range(iseg + 1, end_idx):
                intx_flag = lines_intersect(
                    vtxs[iseg],
                    vtxs[iseg + 1],
                    vtxs[jseg],
                    vtxs[jseg + 1],
                    backend="python",
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

            start_i = endpts[iarc, 0]
            end_i = endpts[iarc, 1]
            start_j = endpts[jarc, 0]
            end_j = endpts[jarc, 1]

            for iseg in range(start_i, end_i):
                for jseg in range(start_j, end_j):
                    intx_flag = lines_intersect(
                        vtxs[iseg],
                        vtxs[iseg + 1],
                        vtxs[jseg],
                        vtxs[jseg + 1],
                        backend="python",
                    )
                    if intx_flag > 0:
                        if iarc < jarc:
                            violations.append((iarc, jarc, iseg, jseg, intx_flag))
                        else:
                            violations.append((jarc, iarc, jseg, iseg, intx_flag))

    return violations
