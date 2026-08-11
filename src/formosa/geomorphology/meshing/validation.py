import numpy as np

import formosa.geomorphology.drainage.network as network_m
from formosa.geomorphology.drainage.network import GraphTopologyError
from formosa.geomorphology.geometry.intersections import IntersectionKind
from formosa.geomorphology.meshing.core import ConstraintKind

from typing import Optional
from numpy.typing import NDArray
from formosa.utils import Backend
from formosa.utils.typing import NpCanonIndex


def _validate_constraints_type_shape(
    indices: NDArray[NpCanonIndex],
    edges: NDArray[NpCanonIndex],
    edge_kinds: NDArray[np.uint8],
):
    if indices.ndim != 2 or indices.shape[1] != 2:
        raise ValueError(
            "'indices' must have shape (V, 2), " + f"but got {indices.shape}."
        )
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("'edges' must have shape (E, 2), " + f"but got {edges.shape}.")
    if edge_kinds.ndim != 1:
        raise ValueError(
            "'edge_kinds' must be a 1D array, " + f"but got {edge_kinds.shape}."
        )
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError("'indices' must contain integers.")
    if not np.issubdtype(edges.dtype, np.integer):
        raise TypeError("'edges' must contain integers.")
    if not np.issubdtype(edge_kinds.dtype, np.integer):
        raise TypeError("'edge_kinds' must contain integer GraphKind flags.")


def _validate_constraints_oob(
    indices: NDArray[NpCanonIndex],
    edges: NDArray[NpCanonIndex],
    shape: Optional[tuple[int, int]] = None,
):
    n_vtxs = indices.shape[0]

    if np.any(edges < 0) or np.any(edges >= n_vtxs):
        invalid = np.flatnonzero(np.any((edges < 0) | (edges >= n_vtxs), axis=1))
        raise GraphTopologyError(
            f"Edges must only reference vertex IDs in [0, {n_vtxs}): "
            + f"but got invalid ID(s) {invalid.tolist()}."
        )
    if np.any(indices < 0):
        invalid = np.flatnonzero(np.any(indices < 0, axis=1))
        raise GraphTopologyError(
            "Raster indices must be non-negative: "
            + f"but got invalid vertex indices {invalid.tolist()}."
        )

    if shape is not None:
        if len(shape) != 2 or shape[0] < 2 or shape[1] < 2:
            raise ValueError("shape must contain at least two rows and two columns.")
        outside = (indices[:, 0] >= shape[0]) | (indices[:, 1] >= shape[1])
        if np.any(outside):
            invalid = np.flatnonzero(outside)
            raise GraphTopologyError(
                f"Vertices must lie inside raster shape {shape},"
                + f"but got vertex indices {invalid.tolist()}."
            )


def _validate_constraints_boundary(
    indices: NDArray[NpCanonIndex],
    edges: NDArray[NpCanonIndex],
    edge_kinds: NDArray[np.uint8],
    shape: tuple[int, int],
):
    nrows, ncols = shape
    perimeter = np.array(
        [(0, j) for j in range(ncols)]
        + [(i, ncols - 1) for i in range(1, nrows)]
        + [(nrows - 1, j) for j in range(ncols - 2, -1, -1)]
        + [(i, 0) for i in range(nrows - 2, 0, -1)],
        dtype=NpCanonIndex,
    )
    exp_bdry = {
        tuple(sorted((tuple(a), tuple(b))))
        for a, b in zip(perimeter, np.roll(perimeter, -1, axis=0))
    }
    bdry_mask = (edge_kinds.astype(np.int64) & int(ConstraintKind.BOUNDARY)) != 0
    bdry = {
        tuple(sorted((tuple(indices[u]), tuple(indices[v]))))
        for u, v in edges[bdry_mask]
    }
    if bdry != exp_bdry:
        missing = sorted(exp_bdry - bdry)
        unexpected = sorted(bdry - exp_bdry)
        raise GraphTopologyError(
            "Boundary constraints do not exactly cover the raster perimeter; "
            f"missing {missing}, unexpected {unexpected}."
        )


def _validate_constraints_intersections(
    indices: NDArray[NpCanonIndex],
    edges: NDArray[NpCanonIndex],
    backend: Backend = "fortran",
):
    # Present every constraint edge as an independent two-vertex arc. The
    # network validator then ignores valid endpoint contacts while reporting
    # crossings, overlaps, T-junctions, and degenerate segments.
    n_edges = edges.shape[0]

    edge_vtxs = indices[edges].reshape(-1, 2)
    starts = np.arange(0, 2 * n_edges, 2, dtype=NpCanonIndex)
    edge_endpts = np.column_stack((starts, starts + 1))
    intxs = network_m.locate_invalid_graph_topology(
        edge_vtxs, edge_endpts, backend=backend
    )
    if intxs is None:
        return
    elif np.all(
        intxs[:, 4]
        == IntersectionKind.DISJOINT_SEGMENTS | intxs[:, 4]
        == IntersectionKind.ENDPOINT_CONTACT
    ):
        return

    intx_names = {
        IntersectionKind.INTERIOR_CROSSING: "interior crossing",
        IntersectionKind.COLLINEAR_OVERLAP: "collinear overlap",
        IntersectionKind.IDENTICAL_SEGMENTS: "identical segment",
        IntersectionKind.T_JUNCTION: "unsplit T-junction",
        IntersectionKind.DEGENERATE_SEGMENT: "degenerate segment",
    }
    details = [
        {
            "edge_ids": (int(record[0]), int(record[1])),
            "type": intx_names.get(
                IntersectionKind(record[4]), f"intersection flag {int(record[4])}"
            ),
        }
        for record in intxs
    ]
    raise GraphTopologyError(
        f"Constraint edges do not form a planar straight-line graph: {details}."
    )


def validate_constraints(
    indices: NDArray[NpCanonIndex],
    edges: NDArray[NpCanonIndex],
    edge_kinds: NDArray[np.uint8],
    shape: Optional[tuple[int, int]] = None,
    backend: Backend = "fortran",
) -> None:
    """
    Validates a normalised constraint graph.
    """
    indices = np.asarray(indices)
    edges = np.asarray(edges)
    edge_kinds = np.asarray(edge_kinds)

    _validate_constraints_type_shape(indices, edges, edge_kinds)
    _validate_constraints_oob(indices, edges, shape)

    n_vtxs = indices.shape[0]
    n_edges = edges.shape[0]

    # Check for normality
    if np.unique(indices, axis=0).shape[0] != n_vtxs:
        raise GraphTopologyError("The constraint graph contains duplicate vertices.")
    if np.any(edges[:, 0] == edges[:, 1]):
        invalid = np.flatnonzero(edges[:, 0] == edges[:, 1])
        raise GraphTopologyError(
            f"Constraint edges cannot be self-edges, "
            + f"but got invalid edges at indices {invalid.tolist()}."
        )
    if np.any(edges[:, 0] > edges[:, 1]):
        invalid = np.flatnonzero(edges[:, 0] > edges[:, 1])
        raise GraphTopologyError(
            "Constraint edge vertex IDs must use canonical increasing order, "
            f"but got invalid edges at indices {invalid.tolist()}."
        )
    if np.unique(edges, axis=0).shape[0] != n_edges:
        raise GraphTopologyError("The constraint graph contains duplicate edges.")

    allowed_kind_bits = int(
        ConstraintKind.VALLEY | ConstraintKind.RIDGE | ConstraintKind.BOUNDARY
    )
    invalid_kinds = (edge_kinds.astype(np.int64) & ~allowed_kind_bits) != 0
    if np.any(invalid_kinds):
        invalid = np.flatnonzero(invalid_kinds)
        raise GraphTopologyError(
            "Edges contain unsupported GraphKind bits at indices "
            f"{invalid.tolist()}."
        )

    if shape is not None:
        _validate_constraints_boundary(indices, edges, edge_kinds, shape)

    _validate_constraints_intersections(indices, edges, backend)
