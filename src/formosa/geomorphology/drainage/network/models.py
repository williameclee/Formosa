"""
Represents drainage networks as flow graphs.

Last modified: 2026-08-18, En-Chi Lee (williameclee@gmail.com)
"""

from dataclasses import dataclass
import numpy as np

from formosa.geomorphology.drainage.network.editing import remove_unused_vertices
from formosa.utils.typing import NpCanonIndex

from typing import Optional
import numpy.typing as npt


@dataclass
class FlowGraph:
    indices: npt.NDArray[NpCanonIndex]
    arc_endpts: npt.NDArray[NpCanonIndex]
    arc_orders: Optional[npt.NDArray[np.int8]] = None

    def __init__(
        self,
        indices: npt.NDArray[np.number],
        endpts: npt.NDArray[np.integer],
        orders: Optional[npt.NDArray[np.integer]] = None,
    ):
        self.indices = indices.astype(NpCanonIndex)
        self.arc_endpts = endpts.astype(NpCanonIndex)
        if orders is not None:
            self.arc_orders = orders.astype(np.int8)

    @property
    def endpts(self) -> npt.NDArray[NpCanonIndex]:
        """Alias for the property `arc_endpts`."""
        return self.arc_endpts

    @property
    def orders(self) -> Optional[npt.NDArray[np.int8]]:
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
