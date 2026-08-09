# Last modified
#   2026-07-12, En-Chi Lee (williameclee@gmail.com)
#     - Implemented Python backend of function
#       `locate_invalid_graph_topology`

from formosa.geomorphology.geometry.intersections import lines_intersect_v2


import numpy as np
import numpy.typing as npt


def locate_invalid_graph_topology(
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
                intx_flag = lines_intersect_v2(
                    vertex_ijs[iseg],
                    vertex_ijs[iseg + 1],
                    vertex_ijs[jseg],
                    vertex_ijs[jseg + 1],
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

            start_i = arc_endpts[iarc, 0]
            end_i = arc_endpts[iarc, 1]
            start_j = arc_endpts[jarc, 0]
            end_j = arc_endpts[jarc, 1]

            for iseg in range(start_i, end_i):
                for jseg in range(start_j, end_j):
                    intx_flag = lines_intersect_v2(
                        vertex_ijs[iseg],
                        vertex_ijs[iseg + 1],
                        vertex_ijs[jseg],
                        vertex_ijs[jseg + 1],
                        backend="python",
                    )
                    if intx_flag > 0:
                        if iarc < jarc:
                            violations.append((iarc, jarc, iseg, jseg, intx_flag))
                        else:
                            violations.append((jarc, iarc, jseg, iseg, intx_flag))

    return violations
