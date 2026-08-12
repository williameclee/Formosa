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
    Creates a rectangular boundary from its 4 corners.

    Additional constraint vertices on the perimeter are inserted
    later while normalising boundary-aligned edges.
    """

    if len(shape) != 2 or shape[0] < 2 or shape[1] < 2:
        raise ValueError("shape must contain at least two rows and two columns.")
    nrows, ncols = shape
    bdry_indices = np.array(
        [(0, 0), (0, ncols - 1), (nrows - 1, ncols - 1), (nrows - 1, 0), (0, 0)],
        dtype=NpCanonIndex,
    )
    return ConstraintInput(
        FlowGraph(
            bdry_indices,
            np.array([[0, bdry_indices.shape[0] - 1]], dtype=NpCanonIndex),
        ),
        kind=ConstraintKind.BOUNDARY,
    )


def _split_boundary_aligned_edges(
    indices: NDArray[NpCanonIndex],
    edges: NDArray[NpCanonIndex],
    edge_kinds: NDArray[np.uint8],
    shape: tuple[int, int],
) -> tuple[NDArray[NpCanonIndex], NDArray[np.uint8]]:
    """
    Splits perimeter edges at every existing perimeter vertex.
    """
    nrows, ncols = shape
    rows = indices[:, 0]
    cols = indices[:, 1]
    sides = {
        "top": np.flatnonzero((rows == 0) & (cols >= 0) & (cols < ncols)),
        "bottom": np.flatnonzero((rows == nrows - 1) & (cols >= 0) & (cols < ncols)),
        "left": np.flatnonzero((cols == 0) & (rows >= 0) & (rows < nrows)),
        "right": np.flatnonzero((cols == ncols - 1) & (rows >= 0) & (rows < nrows)),
    }
    side_axes = {"top": 1, "bottom": 1, "left": 0, "right": 0}
    for name, vertex_ids in sides.items():
        axis = side_axes[name]
        sides[name] = vertex_ids[np.argsort(indices[vertex_ids, axis])]

    split_edges: list[NDArray[NpCanonIndex]] = []
    split_kinds: list[NDArray[np.uint8]] = []
    for edge, kind in zip(edges, edge_kinds):
        a, b = indices[edge]
        side_name: Optional[str] = None
        axis = 0
        if (
            a[0] == b[0]
            and a[0] in (0, nrows - 1)
            and np.all((a[1] >= 0, a[1] < ncols, b[1] >= 0, b[1] < ncols))
        ):
            side_name = "top" if a[0] == 0 else "bottom"
            axis = 1
        elif (
            a[1] == b[1]
            and a[1] in (0, ncols - 1)
            and np.all((a[0] >= 0, a[0] < nrows, b[0] >= 0, b[0] < nrows))
        ):
            side_name = "left" if a[1] == 0 else "right"
            axis = 0

        if side_name is None:
            split_edges.append(edge.reshape(1, 2))
            split_kinds.append(np.array([kind], dtype=np.uint8))
            continue

        side_ids = sides[side_name]
        side_values = indices[side_ids, axis]
        low, high = sorted((int(a[axis]), int(b[axis])))
        chain = side_ids[(side_values >= low) & (side_values <= high)]
        split_edges.append(
            np.column_stack((chain[:-1], chain[1:])).astype(NpCanonIndex)
        )
        split_kinds.append(np.full(chain.size - 1, kind, dtype=np.uint8))

    return np.concatenate(split_edges), np.concatenate(split_kinds)


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
        all_edges = inv_ids[all_edges].astype(NpCanonIndex)
        if np.any(all_edges[:, 0] == all_edges[:, 1]):
            raise GraphTopologyError(
                "An arc contains consecutive vertices at the same raster index."
            )
        if shape is not None:
            all_edges, all_edge_kinds = _split_boundary_aligned_edges(
                indices, all_edges, all_edge_kinds, shape
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
