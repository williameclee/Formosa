"""
Validates flow graphs and report invalid topology.

Last modified: 2026-08-24, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np
from numpy.typing import NDArray

import formosa.geomorphology.drainage.network._backends.validation_py as val_py
from formosa.geomorphology._native import network_validation as val_f
from formosa.geomorphology.drainage.directions import DirectionEncoding
from formosa.geomorphology.drainage.neighbours import (
    compute_downstream_indices,
)
from formosa.utils import Backend, NpCanonIndex, raise_fortran_error
from formosa.utils.typing import NpCoords, NpFlowDir, NpIndex


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

    def __init__(self, cycle_ijs: NDArray[np.integer]) -> None:
        self.cycle_ijs = np.asarray(cycle_ijs, dtype=np.int32).copy()
        super().__init__(
            f"Selected flow graph contains directed cycles at {self.cycle_ijs.tolist()}."
        )


class IncompleteFlowGraphError(GraphTopologyError):
    """
    Raised when construction omits one or more selected directed edges.
    """

    def __init__(
        self,
        missing_vtxs: NDArray[np.integer],
        missing_edges: NDArray[np.integer] | None = None,
    ) -> None:
        self.missing_ijs = np.asarray(missing_vtxs, dtype=np.int32).copy()
        if missing_edges is None:
            self.missing_edges = np.empty((0, 4), dtype=np.int32)
        else:
            self.missing_edges = np.asarray(missing_edges, dtype=np.int32).copy()
        super().__init__(
            f"Flow-graph construction omitted selected directed edges {self.missing_edges.tolist()}; "
            + f"participating cells are {self.missing_ijs.tolist()}."
        )


def _valid_flow_edges(
    dirs: NDArray[NpFlowDir], valids: NDArray[np.bool_], dir_enc: DirectionEncoding
) -> tuple[NDArray[NpCanonIndex], NDArray[NpCanonIndex], NDArray[np.bool_]]:
    """
    Returns downstream indices and a mask indicating whether the
    cell flows into a valid neighbouring (non-self) edge.
    """
    dsi, dsj, _, ds_inbounds = compute_downstream_indices(
        dirs, dir_enc, check=False, return_flat_index=False, oob_is_okay=True
    )

    # Whether the downstream cell is also valid (not just inbound)
    ds_valids = np.zeros(dirs.shape, dtype=bool)
    ds_valids[ds_inbounds] = valids[dsi[ds_inbounds], dsj[ds_inbounds]]

    # Exclude self-loops (where offsets di, dj == 0)
    not_self = dirs != dir_enc.no_flow_code

    has_valid_ds = valids & ds_valids & not_self
    return dsi, dsj, has_valid_ds


def _validate_flowgraph_coverage(
    vtxs: NDArray[NpIndex],
    arc_endpts: NDArray[NpIndex],
    dsi: NDArray[NpIndex],
    dsj: NDArray[NpIndex],
    has_valid_ds: NDArray[np.bool_],
) -> None:
    """
    Checks that every selected directed edge occurs in a returned
    graph arc.
    """
    represented = np.zeros(has_valid_ds.shape, dtype=bool)

    # Identify consecutive vertex pairs that belong to an arc.
    seg_cnts = np.zeros(vtxs.shape[0], dtype=np.int32)
    np.add.at(seg_cnts, arc_endpts[:, 0], 1)
    np.add.at(seg_cnts, arc_endpts[:, 1], -1)
    seg_valids = np.cumsum(seg_cnts)[:-1] > 0

    srcs = vtxs[:-1][seg_valids]
    targets = vtxs[1:][seg_valids]

    # Confirm each represented edge matches the source cell's expected downstream.
    matches = (targets[:, 0] == dsi[srcs[:, 0], srcs[:, 1]]) & (
        targets[:, 1] == dsj[srcs[:, 0], srcs[:, 1]]
    )
    matched_srcs = srcs[matches]
    represented[matched_srcs[:, 0], matched_srcs[:, 1]] = True

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
    vtxs: NDArray[NpCoords],
    endpts: NDArray[NpIndex],
) -> NDArray[np.int32] | None:
    """
    Returns every topology violation using the capacity-aware
    Fortran scanner.

    The first scan uses a small provisional output buffer. If the
    exact count reported by that scan exceeds the buffer, the scan
    is repeated with a buffer of exactly the required size.
    Incomplete provisional results are never returned.

    Parameters
    ----------
    vtxs : NDArray[number]
        Vertex coordinates with shape `(V,2)`.
    endpts : NDArray[integer]
        Inclusive, 0-based arc endpoint indices with shape `(A,2)`.

    Returns
    -------
    NDArray[int32] or None
        Complete `(nintxs, 5)` intersection records using 0-based
        indices, or `None` when no violations are found.
    """
    vtxs_f = np.asfortranarray(vtxs.T, dtype=np.float32)
    endpts_f = np.asfortranarray(endpts.T, dtype=np.int32) + 1
    cpty = max(vtxs_f.shape[1] // 100, 3)  # Arbitrary capacity that seems to work

    intxs, nintxs, err_code = val_f.scan_invalid_graph_topology(vtxs_f, endpts_f, cpty)
    raise_fortran_error("scan_invalid_graph_topology", err_code)

    if nintxs == 0:
        return None

    if nintxs > cpty:
        exp_nintxs = nintxs
        intxs, nintxs, err_code = val_f.scan_invalid_graph_topology(
            vtxs_f, endpts_f, exp_nintxs
        )
        raise_fortran_error("scan_invalid_graph_topology", err_code)
        if nintxs != exp_nintxs:
            raise RuntimeError(
                "Topology-intersection count changed during exact-size retry."
            )

    intxs = intxs[:, :nintxs]
    intxs[:-1, :] -= 1  # Convert to 0-based indexing, except the intersection flag
    return intxs.T.astype(np.int32, order="C")


def locate_invalid_graph_topology(
    vtxs: NDArray[NpCoords], endpts: NDArray[NpIndex], backend: Backend = "fortran"
) -> NDArray[np.int32] | None:
    """
    Locates invalid topologies (segment intersections) within and
    between arcs in a graph.

    This function checks for self-intersections within individual
    arcs, as well as intersections between segments of different
    arcs. The intersection checks are performed using a 2D line
    segment intersection algorithm.

    Parameters
    ----------
    vtxs : NDArray[number]
        Grid coordinates (i, j) of each vertex.
        - Expected shape: `(nvtxs, 2)`.
    endpts : NDArray[int]
        Start and end vertex indices for each arc in `vtxs`.
        - Expected shape: `(narcs, 2)`.
    backend : {'fortran', 'python'}, optional
        Backend to use for computation.
        `'fortran'` uses the Fortran extension for performance,
        while `'python'` uses a pure Python implementation.
        - Default backend is `'fortran'`.

    Returns
    -------
    intxs : NDArray[int32] | None
        Detected intersections, or `None` if no intersections are
        found.
        - Shape: `(nintxs, 5)`.
        The rows are sorted lexicographically and each row contains:
        - `iarc`: Index of the first arc (0-based).
        - `jarc`: Index of the second arc (0-based).
        - `iseg`: Start vertex index of the first intersecting
            segment (0-based).
        - `jseg`: Start vertex index of the second intersecting
            segment (0-based).
        - `intx_flag`: Flag indicating the type of intersection:
            - 1 : Interior-interior crossing (X).
            - 2 : Collinear overlap, not identical.
            - 3 : Identical segment.
            - 4 : Endpoint-on-interior (T-junction).
            - 5 : Degenerate segment (some line is actually a point).
    """
    if vtxs.ndim != 2 or vtxs.shape[1] != 2:
        raise ValueError("Invalid array shapes passed.")
    elif endpts.ndim != 2 or endpts.shape[1] != 2:
        raise ValueError("Invalid array shapes passed.")

    match backend:
        case "python":
            intxs = val_py.locate_invalid_graph_topology(
                endpts.astype(np.int32, order="C"),
                vtxs.astype(np.float64, order="C"),
            )
            if not intxs:
                return None
            intxs = np.array(intxs, dtype=np.int32, order="C")
        case "fortran":
            intxs = _locate_invalid_graph_topology_fortran(vtxs, endpts)
            if intxs is None:
                return None
    if intxs.shape[0] > 1:
        # Sort lexicographically
        sort_idx = np.lexsort((intxs[:, 3], intxs[:, 2], intxs[:, 1], intxs[:, 0]))
        intxs = intxs[sort_idx]
    return intxs


def _ignore_identical_intergraph_arcs(
    intxs: NDArray[np.int32] | None,
    vtxs: NDArray[NpCoords],
    endpts: NDArray[NpIndex],
    graph_ids: NDArray[np.integer],
) -> NDArray[np.int32] | None:
    """
    Removes topology violations between identical arcs in different
    graphs.
    """
    if intxs is None:
        return None

    keeps = np.ones(intxs.shape[0], dtype=bool)
    identical_pairs: dict[tuple[int, int], bool] = {}
    for i, (iarc, jarc, _, _, _) in enumerate(intxs):
        if graph_ids[iarc] == graph_ids[jarc]:
            continue

        pair = (int(iarc), int(jarc))
        if pair not in identical_pairs:
            istart, iend = endpts[:, iarc]
            jstart, jend = endpts[:, jarc]
            iarc_vtxs = vtxs[:, istart : iend + 1]
            jarc_vtxs = vtxs[:, jstart : jend + 1]
            identical_pairs[pair] = np.array_equal(
                iarc_vtxs, jarc_vtxs
            ) or np.array_equal(iarc_vtxs, jarc_vtxs[:, ::-1])
        if identical_pairs[pair]:
            keeps[i] = False

    if not np.any(keeps):
        return None
    return intxs[keeps]


def _locate_disallowed_graph_topology(
    vtxs: NDArray[NpCoords],
    endpts: NDArray[NpIndex],
    graph_ids: NDArray[np.integer] | None = None,
) -> NDArray[np.int32] | None:
    """
    Locates violations in arrays stored in internal (2,N) layout.
    """
    intxs = locate_invalid_graph_topology(vtxs.T, endpts.T)
    if graph_ids is not None:
        intxs = _ignore_identical_intergraph_arcs(intxs, vtxs, endpts, graph_ids)
    return intxs
