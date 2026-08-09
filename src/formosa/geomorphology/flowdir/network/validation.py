# Last modified
#   2026-07-12, En-Chi Lee (williameclee@gmail.com)
#     - Implemented Python and FORTRAN backends of function 
#       `locate_invalid_graph_topology`
#   2026-07-14, En-Chi Lee (williameclee@gmail.com)
#     - Updated variable names in `locate_invalid_graph_topology`
#   2026-07-29, En-Chi Lee (williameclee@gmail.com)
#     - Made topology intersection results complete using scan-and-
#       retry

import numpy as np

from typing import Literal, Optional
import numpy.typing as npt

from formosa.geomorphology.flowdir.directions import D8Directions
from formosa.geomorphology.flowdir.utils import (
    compute_downstream_indices,
    raise_fortran_error,
)
from formosa.geomorphology.flowdir_f import flowdir_graphs as graphs_f


class GraphTopologyError(RuntimeError):
    """
    Base exception for a graph that fails topology validation.
    """


class InvalidOriginalGraphTopology(GraphTopologyError):
    """
    Raised when an invalid result originated from invalid input topology.
    """


class UnresolvedSimplificationTopology(GraphTopologyError):
    """
    Raised when simplification leaves invalid topology from valid input.
    """


class DirectedFlowCycleError(GraphTopologyError):
    """
    Raised when the selected flow field contains one or more directed cycles.
    """

    def __init__(self, cycle_ijs: npt.NDArray[np.integer]) -> None:
        self.cycle_ijs = np.asarray(cycle_ijs, dtype=np.int32).copy()
        super().__init__(
            "Selected flow graph contains directed cycles at "
            f"{self.cycle_ijs.tolist()}."
        )


class IncompleteFlowGraphError(GraphTopologyError):
    """
    Raised when construction omits one or more selected directed edges.
    """

    def __init__(
        self,
        missing_ijs: npt.NDArray[np.integer],
        missing_edges: Optional[npt.NDArray[np.integer]] = None,
    ) -> None:
        self.missing_ijs = np.asarray(missing_ijs, dtype=np.int32).copy()
        if missing_edges is None:
            self.missing_edges = np.empty((0, 4), dtype=np.int32)
        else:
            self.missing_edges = np.asarray(missing_edges, dtype=np.int32).copy()
        super().__init__(
            "Flow-graph construction omitted selected directed edges "
            f"{self.missing_edges.tolist()}; participating cells are "
            f"{self.missing_ijs.tolist()}."
        )


def _valid_flow_edges(
    dirs: npt.NDArray[np.integer],
    valids: npt.NDArray[np.bool_],
    dir_scheme: D8Directions,
) -> tuple[
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.bool_],
]:
    """
    Returns downstream indices and a mask indicating whether the cell flows into a valid neighbouring (non-self) edge.
    """
    dsi, dsj, _, ds_inbounds = compute_downstream_indices(
        dirs,
        dir_scheme=dir_scheme,
        check=False,
        return_flat_index=False,
        oob_is_okay=True,
    )

    # Whether the downstream cell is also valid (not just inbound)
    ds_valids = np.zeros(dirs.shape, dtype=bool)
    ds_valids[ds_inbounds] = valids[dsi[ds_inbounds], dsj[ds_inbounds]]

    # Exclude self-loops (where offsets di, dj == 0)
    not_self = dirs != dir_scheme.no_flow_code

    has_valid_ds = valids & ds_valids & not_self
    return dsi, dsj, has_valid_ds


def _validate_flowgraph_coverage(
    vertex_ijs: npt.NDArray[np.integer],
    arc_endpts: npt.NDArray[np.integer],
    dsi: npt.NDArray[np.integer],
    dsj: npt.NDArray[np.integer],
    has_valid_ds: npt.NDArray[np.bool_],
) -> None:
    """
    Checks that every selected directed edge occurs in a returned graph arc.
    """
    represented = np.zeros(has_valid_ds.shape, dtype=bool)

    # Identify consecutive vertex pairs that belong to an arc.
    segment_counts = np.zeros(vertex_ijs.shape[0], dtype=np.int32)
    np.add.at(segment_counts, arc_endpts[:, 0], 1)
    np.add.at(segment_counts, arc_endpts[:, 1], -1)
    segment_valids = np.cumsum(segment_counts)[:-1] > 0

    sources = vertex_ijs[:-1][segment_valids]
    targets = vertex_ijs[1:][segment_valids]

    # Confirm each represented edge matches the source cell's expected downstream.
    matches = (targets[:, 0] == dsi[sources[:, 0], sources[:, 1]]) & (
        targets[:, 1] == dsj[sources[:, 0], sources[:, 1]]
    )
    matched_sources = sources[matches]
    represented[matched_sources[:, 0], matched_sources[:, 1]] = True

    missing_sources = np.argwhere(has_valid_ds & ~represented)
    if missing_sources.size:
        missing_targets = np.column_stack(
            (
                dsi[missing_sources[:, 0], missing_sources[:, 1]],
                dsj[missing_sources[:, 0], missing_sources[:, 1]],
            )
        )
        missing_edges = np.column_stack((missing_sources, missing_targets))
        missing_ijs = np.unique(missing_edges.reshape(-1, 2), axis=0)
        raise IncompleteFlowGraphError(missing_ijs, missing_edges)


def _locate_invalid_graph_topology_fortran(
    vertex_xys: npt.NDArray[np.number],
    arc_endpts: npt.NDArray[np.integer],
) -> Optional[npt.NDArray[np.int32]]:
    """
    Returns every topology violation using the capacity-aware FORTRAN scanner.

    The first scan uses a small provisional output buffer. If the exact count
    reported by that scan exceeds the buffer, the scan is repeated with a
    buffer of exactly the required size. Incomplete provisional results are
    never returned.

    Parameters
    ----------
    vertex_xys : NDArray[number]
        Vertex coordinates with shape `(nvertices, 2)`.
    arc_endpts : NDArray[integer]
        Inclusive, zero-based arc endpoint indices with shape `(narcs, 2)`.

    Returns
    -------
    NDArray[int32] or None
        Complete `(nintxs, 5)` intersection records using zero-based indices,
        or `None` when no violations are found.

    Raises
    ------
    ValueError
        If the low-level scanner rejects its inputs.
    MemoryError
        If scanner workspace or result allocation fails.
    RuntimeError
        If the scanner returns an unexpected status or the exact count changes
        during the retry.
    """
    vertices_f = np.asfortranarray(vertex_xys.T, dtype=np.float32)
    endpts_f = np.asfortranarray(arc_endpts.T, dtype=np.int32) + 1
    capacity = max(
        vertices_f.shape[1] // 100, 3
    )  # Arbitrary capacity that seems to work

    intxs, nintxs, err_code = graphs_f.scan_invalid_graph_topology(
        vertices_f, endpts_f, capacity
    )
    raise_fortran_error(
        "scan_invalid_graph_topology",
        err_code,
    )

    if nintxs == 0:
        return None

    if nintxs > capacity:
        expected_nintxs = nintxs
        intxs, nintxs, err_code = graphs_f.scan_invalid_graph_topology(
            vertices_f, endpts_f, expected_nintxs
        )
        raise_fortran_error(
            "scan_invalid_graph_topology",
            err_code,
        )
        if nintxs != expected_nintxs:
            raise RuntimeError(
                "Topology-intersection count changed during exact-size retry."
            )

    intxs = intxs[:, :nintxs]
    intxs[:-1, :] -= 1  # Convert to 0-based indexing, except the intersection flag
    return intxs.T.astype(np.int32, order="C")


def locate_invalid_graph_topology(
    vertex_xys: npt.NDArray[np.number],
    arc_endpts: npt.NDArray[np.integer],
    backend: Literal["fortran", "python"] = "fortran",
) -> Optional[npt.NDArray[np.int32]]:
    """
    Locates invalid topologies (segment intersections) within and between arcs in a graph.

    This function checks for self-intersections within individual arcs, as well as intersections between segments of different arcs. The intersection checks are performed using a 2D line segment intersection algorithm.

    Parameters
    ----------
    vertex_xys : NDArray[number]
        2D array of shape `(nvertices, 2)` representing the grid coordinates (i, j) of each vertex.
    arc_endpts : NDArray[integer]
        2D array of shape `(narcs, 2)` containing the start and end vertex indices for each arc in `vertex_ijs`.
    backend : {'fortran', 'python'}, optional
        Computational backend to use.
        The default option is `'fortran'`.

    Returns
    -------
    NDArray[int32] or None
        2D array of shape `(nintxs, 5)` representing the detected intersections, or `None` if no intersections are found.
        The rows are sorted lexicographically and each row contains:
        - `iarc`: Index of the first arc (0-based).
        - `jarc`: Index of the second arc (0-based).
        - `iseg`: Start vertex index of the first intersecting segment (0-based).
        - `jseg`: Start vertex index of the second intersecting segment (0-based).
        - `intx_flag`: Flag indicating the type of intersection:
            - 1 : Interior-interior crossing (X).
            - 2 : Collinear overlap, not identical.
            - 3 : Identical segment.
            - 4 : Endpoint-on-interior (T-junction).
            - 5 : Degenerate segment (some line is actually a point).

    Raises
    ------
    ValueError
        If the shape of `vertex_ijs` or `arc_endpts` is invalid.
    MemoryError
        If the FORTRAN backend cannot allocate its scan workspace or result.
    RuntimeError
        If the FORTRAN scanner returns an unexpected error or an inconsistent
        count during the exact-size retry.
    """
    if vertex_xys.ndim != 2 or vertex_xys.shape[1] != 2:
        raise ValueError("Invalid array shapes passed.")
    elif arc_endpts.ndim != 2 or arc_endpts.shape[1] != 2:
        raise ValueError("Invalid array shapes passed.")

    match backend:
        case "python":
            from ._backends.graphs_py import _locate_invalid_graph_topology_py

            intxs = _locate_invalid_graph_topology_py(
                arc_endpts.astype(np.int32, order="C"),
                vertex_xys.astype(np.float64, order="C"),
            )
            if not intxs:
                return None
            intxs = np.array(intxs, dtype=np.int32, order="C")
        case "fortran":
            intxs = _locate_invalid_graph_topology_fortran(vertex_xys, arc_endpts)
            if intxs is None:
                return None
    if intxs.shape[0] > 1:
        # Sort lexicographically
        sort_idx = np.lexsort((intxs[:, 3], intxs[:, 2], intxs[:, 1], intxs[:, 0]))
        intxs = intxs[sort_idx]
    return intxs
