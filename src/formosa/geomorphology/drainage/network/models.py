"""
Represents drainage networks as flow graphs.

Last modified: 2026-08-18, En-Chi Lee (williameclee@gmail.com)
"""

from dataclasses import dataclass
import numpy as np

from formosa.geomorphology.drainage.network.editing import (
    remove_unused_vertices,
    concat_flowgraph,
)
from formosa.geomorphology.drainage.network.simplification import (
    simplify_flowgraph,
)
from formosa.utils import Backend

from formosa.utils.typing import NpCanonIndex
from numpy.typing import NDArray


@dataclass
class FlowGraph:
    indices: NDArray[NpCanonIndex]
    arc_endpts: NDArray[NpCanonIndex]
    arc_orders: NDArray[np.int8]

    def __init__(
        self,
        indices: NDArray[np.number],
        endpts: NDArray[np.integer],
        orders: NDArray[np.integer],
    ):
        self.indices = indices.astype(NpCanonIndex)
        self.arc_endpts = endpts.astype(NpCanonIndex)
        self.arc_orders = orders.astype(np.int8)

    @property
    def endpts(self) -> NDArray[NpCanonIndex]:
        """Alias for the property `arc_endpts`."""
        return self.arc_endpts

    @property
    def orders(self) -> NDArray[np.int8]:
        """Alias for the property `arc_orders`."""
        return self.arc_orders

    @property
    def n_vtxs(self) -> int:
        """Number of (un-normalised) vertices in the graph."""
        return self.indices.shape[0]

    @property
    def n_arcs(self) -> int:
        """Number of (un-normalised) arcs in the graph."""
        return self.arc_endpts.shape[0]

    def cleanup(self) -> "FlowGraph":
        self.indices, self.arc_endpts = remove_unused_vertices(
            self.indices, self.arc_endpts
        )
        return self

    def concat(self) -> "FlowGraph":
        self.arc_orders, self.indices, self.arc_endpts = concat_flowgraph(
            self.arc_orders, self.indices, self.arc_endpts
        )
        return self

    def simplify(
        self,
        tol: int | float = 1,
        check_topology: bool = True,
        remove_unused: bool = False,
        backend: Backend = "fortran",
    ) -> "FlowGraph":
        self.arc_orders, self.indices, self.arc_endpts, _ = simplify_flowgraph(
            *(self.arc_orders, self.indices, self.arc_endpts),
            tol=tol,
            check_topology=check_topology,
            remove_unused=remove_unused,
            backend=backend,
        )
        return self

    def plot(self, lw=lambda o: o * 0.25, **kwargs) -> None:
        import matplotlib.pyplot as plt

        graph = self.concat()
        for iorder, order in enumerate(graph.orders):
            plt.plot(
                graph.indices[graph.endpts[iorder, 0] : graph.endpts[iorder, 1] + 1, 0],
                graph.indices[graph.endpts[iorder, 0] : graph.endpts[iorder, 1] + 1, 1],
                lw=lw(order),
                **kwargs,
            )
