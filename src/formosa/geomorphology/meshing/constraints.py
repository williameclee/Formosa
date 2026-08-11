from dataclasses import dataclass
import numpy as np

from formosa.geomorphology.drainage.network import GraphTopologyError
from formosa.geomorphology.drainage.network.models import FlowGraph
from formosa.geomorphology.meshing.core import ConstraintKind
from formosa.geomorphology.meshing.validation import validate_constraints

from typing import Iterable, Optional
from numpy.typing import NDArray
from formosa.utils import Backend
from formosa.utils.typing import NpCanonIndex


@dataclass(frozen=True)
class ConstraintInput:
    graph: FlowGraph
    kind: ConstraintKind


def _make_boundary_constraints(shape: tuple[int, int]) -> ConstraintInput:
    """
    Creates boundary constraints based on the given raster shape.
    """

    if len(shape) != 2 or shape[0] < 2 or shape[1] < 2:
        raise ValueError("shape must contain at least two rows and two columns")
    nrows, ncols = shape
    bdry_indices = np.array(
        [(0, j) for j in range(ncols)]
        + [(i, ncols - 1) for i in range(1, nrows)]
        + [(nrows - 1, j) for j in range(ncols - 2, -1, -1)]
        + [(i, 0) for i in range(nrows - 2, 0, -1)],
        dtype=NpCanonIndex,
    )
    # Repeat the first perimeter vertex to represent the closing edge.
    bdry_indices = np.vstack((bdry_indices, bdry_indices[0]))
    return ConstraintInput(
        FlowGraph(
            bdry_indices,
            np.array([[0, bdry_indices.shape[0] - 1]], dtype=NpCanonIndex),
        ),
        kind=ConstraintKind.BOUNDARY,
    )


@dataclass
class ConstraintGraph:
    indices: NDArray[NpCanonIndex]
    edges: NDArray[NpCanonIndex]
    edge_kinds: NDArray[np.uint8]

    def __init__(
        self,
        constraints: ConstraintInput | Iterable[ConstraintInput],
        shape: Optional[tuple[int, int]] = None,
    ):
        if isinstance(constraints, ConstraintInput):
            constraints = [constraints]
        else:
            constraints = list(constraints)
            if len(constraints) == 0 and shape is None:
                raise ValueError("No graphs provided.")

        if shape is not None:
            constraints.append(_make_boundary_constraints(shape))

        constraints = [
            ConstraintInput(
                FlowGraph(
                    cstr.graph.indices, cstr.graph.endpts, cstr.graph.orders
                ).cleanup(),
                kind=cstr.kind,
            )
            for cstr in constraints
        ]  # Don't change the input graphs
        all_indices = np.concat([cstr.graph.indices for cstr in constraints], axis=0)
        indices_offsets = np.concat(
            (
                np.array([0], dtype=np.int32),
                np.cumsum(np.array([cstr.graph.n_vtxs for cstr in constraints])),
            )
        )
        all_endpts = np.concat(
            [
                cstr.graph.endpts + indices_offsets[i]
                for i, cstr in enumerate(constraints)
            ],
            axis=0,
        )
        all_edge_list = []
        all_edge_kind_list = []
        arc_kinds = np.concat(
            [
                np.full(cstr.graph.n_arcs, int(cstr.kind), dtype=np.uint8)
                for cstr in constraints
            ]
        )
        for iarc in range(all_endpts.shape[0]):
            # Skip 0- or invalid-length arcs
            if all_endpts[iarc, 1] == all_endpts[iarc, 0]:
                continue
            elif all_endpts[iarc, 1] < all_endpts[iarc, 0]:
                raise GraphTopologyError(
                    "Ending endpoint of an arc must come later than the starting endpoint in the vertex array, "
                    + f"but got start = {all_endpts[iarc, 0]}, end = {all_endpts[iarc, 1]}."
                )
            elif (all_endpts[iarc, 1] >= all_indices.shape[0]) or (
                all_endpts[iarc, 0] < 0
            ):
                raise GraphTopologyError(
                    "Attempting to reference an out-of-bound vertex: "
                    + f"vertex array capacity is [{0, np.size(all_indices,0)}], "
                    + f"but try to get vertices [{all_endpts[iarc, 0]}, {all_endpts[iarc, 1]}]"
                )
            first = np.arange(
                all_endpts[iarc, 0], all_endpts[iarc, 1], dtype=NpCanonIndex
            )
            all_edge_list.append(np.column_stack((first, first + 1)))
            all_edge_kind_list.append(
                np.full(first.size, arc_kinds[iarc], dtype=np.uint8)
            )
        all_edges = (
            np.concat(all_edge_list, axis=0)
            if all_edge_list
            else np.empty((0, 2), dtype=NpCanonIndex)
        )
        all_edge_kinds = (
            np.concat(all_edge_kind_list)
            if all_edge_kind_list
            else np.empty(0, dtype=np.uint8)
        )

        # Deduplicate
        indices, inv_ids = np.unique(all_indices, axis=0, return_inverse=True)
        all_edges = inv_ids[all_edges]
        if np.any(all_edges[:, 0] == all_edges[:, 1]):
            raise GraphTopologyError(
                "An arc contains consecutive vertices at the same raster index."
            )
        all_edges.sort(axis=1)
        edges, edge_inv_ids = np.unique(all_edges, axis=0, return_inverse=True)
        edge_kinds = np.zeros(edges.shape[0], dtype=np.uint8)
        np.bitwise_or.at(edge_kinds, edge_inv_ids, all_edge_kinds)
        self.indices = indices.astype(NpCanonIndex, copy=False)
        self.edges = edges.astype(NpCanonIndex, copy=False)
        self.edge_kinds = edge_kinds

        self.validate(shape)

    def validate(
        self, shape: Optional[tuple[int, int]] = None, backend: Backend = "fortran"
    ) -> None:
        validate_constraints(self.indices, self.edges, self.edge_kinds, shape, backend)
