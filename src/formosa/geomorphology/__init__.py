# Last modified
#   2026-02-11, En-Chi Lee (williameclee@arizona.edu)
#     - Rename flowdir functions to be more descriptive
#   2026-06-09, En-Chi Lee (williameclee@gmail.com)
#     - Added `compute_flow_dist2ridge` function to compute 'distance to ridges'

from formosa.geomorphology.d8directions import D8Directions
from formosa.geomorphology.terrain import compute_slope
from formosa.geomorphology.flowdir import (
    get_neighbour_values,
    fill_depressions,
    compute_flowdir,
    create_flowgraph,
    count_indegree,
    compute_flow_accumulation,
    compute_flow_strahler_order,
    compute_dist2source,
    compute_dist2sink,
    label_watersheds,
    compute_dist2conf_max,
    compute_dist2ridge,
)
