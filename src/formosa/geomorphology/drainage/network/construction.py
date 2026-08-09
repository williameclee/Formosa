"""
Conversions of a flow direction raster to a flow graph.

Last modified: 2026-07-09, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np

from formosa.utils import Backend, raise_fortran_error
from formosa.geomorphology.drainage.directions import D8Directions
from formosa.geomorphology.drainage.neighbours import (
    compute_downstream_indices,
)
import formosa.geomorphology.drainage.flowdir as flowdir_m
import formosa.geomorphology.drainage.metrics as metrics_m
from formosa.geomorphology.drainage.network.editing import remove_unused_vertices
from formosa.geomorphology.drainage.network.validation import (
    DirectedFlowCycleError,
    _valid_flow_edges,
    _validate_flowgraph_coverage,
)
from formosa.geomorphology._native import network_construction as constr_f
import formosa.geomorphology.drainage.network._backends.construction_py as constr_py

from typing import Optional
import numpy.typing as npt


def create_flowgraph(
    dirs: npt.NDArray[np.integer],
    valids: Optional[npt.NDArray[np.bool_]] = None,
    dir_scheme: D8Directions = D8Directions(),
    x: Optional[npt.NDArray[np.number]] = None,
    y: Optional[npt.NDArray[np.number]] = None,
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]]:
    """
    Computes a graph representation of the flow directions in a flow direction grid.

    Parameters
    ----------
    dirs : NDArray[int]
        2D array representing the flow directions for each cell.
    valids : NDArray[bool], optional
        Boolean mask array indicating valid cells in the flow direction grid.
        If `None`, all cells are considered valid.
        Default is `None`.
    directions : D8Directions, optional
        Instance of `D8Directions` defining the flow direction scheme.
        Default is `D8Directions()`.
    x : NDArray[number], optional
        2D array representing the x-coordinates of each cell.
        If provided, the graph will use these coordinates instead of grid indices.
        Default is `None`.
    y : NDArray[number], optional
        2D array representing the y-coordinates of each cell.
        If provided, the graph will use these coordinates instead of grid indices.
        Default is `None`.

    Returns
    -------
    graphi : NDArray[int]
        1D array representing the row indices of the graph edges.
    graphj : NDArray[int]
        1D array representing the column indices of the graph edges.
    """
    if valids is not None:
        assert (
            valids.shape == dirs.shape
        ), f"Shape for dlowdirs and valids mask must match, but got valid shape {dirs.shape} and flowdirs shape {valids.shape} instead"
    else:
        valids = np.full(dirs.shape, True, dtype=bool)

    i, j = np.meshgrid(
        np.arange(dirs.shape[0], dtype=np.int32),
        np.arange(dirs.shape[1], dtype=np.int32),
        indexing="ij",
    )
    dsi, dsj, _, ds_valids = compute_downstream_indices(
        dirs, dir_scheme=dir_scheme, check=False, return_flat_index=False
    )

    if x is not None and y is not None:
        j, i = x, y

        # Map i,j to actual coordinates
        dsx = np.full_like(dsj, np.nan, dtype=np.float64)
        dsy = np.full_like(dsj, np.nan, dtype=np.float64)
        dsx[ds_valids] = x[dsi[ds_valids], dsj[ds_valids]]
        dsy[ds_valids] = y[dsi[ds_valids], dsj[ds_valids]]
        dsi, dsj = dsy, dsx

    graphi = np.stack(
        (
            i[valids & ds_valids],
            dsi[valids & ds_valids],
            np.full(i[valids & ds_valids].size, np.nan),
        ),
        axis=1,
    ).ravel(order="C")
    graphj = np.stack(
        (
            j[valids & ds_valids],
            dsj[valids & ds_valids],
            np.full(j[valids & ds_valids].size, np.nan),
        ),
        axis=1,
    ).ravel(order="C")
    return graphi, graphj


def construct_flowgraph(
    dirs: npt.NDArray[np.integer],
    dir_scheme: D8Directions = D8Directions(),
    valids: Optional[npt.NDArray[np.bool_]] = None,
    min_order: int = 2,
    orders: Optional[npt.NDArray[np.integer]] = None,
    preserve_junctions: bool = True,
    sort: bool = True,
    remove_unused: bool = False,
    backend: Backend = "fortran",
) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """
    Constructs a flow graph from a flow direction grid.

    Parameters
    ----------
    dirs : NDArray[int], optional
        2D array representing the flow directions for each cell.
    dir_scheme : D8Directions, optional
        Instance of `D8Directions` defining the flow direction scheme.
        Default scheme is `D8Directions()`.
    valids : NDArray[bool], optional
        Boolean mask array indicating valid cells in the flow direction grid.
        If `None`, all cells are considered valid.
        Default mask is `None`.
    min_order : int, optional
        Minimum Strahler order to include in the flow graph (see `orders`).
        Default order is 2.
    orders : NDArray[uint8], optional
        2D integer array representing the Strahler order for each cell.
        If `None`, it will be computed from the flow direction grid.
        Default input is `None`.
    preserve_junctions : bool, optional
        Whether to preserve junctions in the flow graph.
        Default option is `True`.
    sort : bool, option
        Whether to sort the flow graph by arc order and then by length.
        Default option is `True`.
    remove_unused : bool, optional
        Whether to compact the vertex array after construction so the arc ranges are adjacent in arc order.
        Default option is `False`.
    backend : {'fortran', 'python'}, optional
        Backend to use for computation.
        `'fortran'` uses the FORTRAN extension for performance,
        while `'python'` uses a pure Python implementation.
        Default backend is `'fortran'`.

    Returns
    -------
    arc_orders : NDArray[int8]
        1D array representing the Strahler order for each arc in the flow graph.
    vertex_ijs : NDArray[int32]
        V-by-2 array containing the ordered (i, j) incices of all arcs, concactinated together.
    vertex_endpts : NDArray[int32]
        A-by-2 array containing the indices of where each arc starts and ends in `vertex_ijs`.
        The returned endpoints are inclusive, meaning slicing must be done as `vertex_ijs[start : end + 1]`.

    Raises
    ------
    DirectedFlowCycleError
        If the selected flow field contains a directed cycle.
    IncompleteFlowGraphError
        If either endpoint of a selected directed edge is absent from the
        constructed graph.

    Notes
    -----
    Selected cells with no selected incoming or outgoing edge are intentionally
    omitted from the arc representation.
    """
    if valids is None:
        valids = np.ones(dirs.shape, dtype=bool)
    if orders is None:
        orders = metrics_m.compute_flow_strahler_order(
            dirs,
            dir_scheme=dir_scheme,
            valids=valids,
            backend=backend,
        )

    # Find seed cells to start with
    valids = valids & (orders >= min_order)
    ncells = int(np.sum(valids))
    indegs = flowdir_m.count_indegree(
        dirs, dir_scheme=dir_scheme, valids=valids, backend=backend
    )
    cyclics = flowdir_m.find_cyclic_flowdirs(
        dirs,
        dir_scheme=dir_scheme,
        valids=valids,
        indegs=indegs,
        backend=backend,
    )
    cycle_ijs = np.argwhere(cyclics).astype(np.int32, order="C")
    if cycle_ijs.size > 0:
        raise DirectedFlowCycleError(cycle_ijs)

    seeds = valids & (indegs == 0)

    match backend:
        case "python":
            narcs, nvertices, arc_orders, vertex_ijs, arc_endpts = (
                constr_py.construct_flowgraph(
                    dirs=dirs,
                    dir_scheme=dir_scheme,
                    valids=valids,
                    orders=orders,
                    indegs=indegs,
                    seeds=seeds,
                    preserve_junctions=preserve_junctions,
                    ncells=ncells,
                )
            )
        case "fortran":
            narcs, nvertices, arc_orders, vertex_ijs, arc_endpts, err_code = (
                constr_f.construct_flowgraph(
                    dirs.astype(np.uint8, order="F"),
                    valids.astype(bool, order="F"),
                    orders.astype(np.int16, order="F"),
                    seeds.astype(np.bool_, order="F"),
                    indegs.astype(np.int8, order="F"),
                    dir_scheme.offsets.astype(np.int32, order="F"),
                    dir_scheme.codes.astype(np.uint8, order="F"),
                    preserve_junctions,
                    ncells,
                )
            )
            raise_fortran_error("construct_flowgraph", err_code)
            # Convert from 1-based index to 0-based index
            vertex_ijs -= 1
            arc_endpts -= 1

    arc_orders = arc_orders[:narcs].T.copy(order="C")
    arc_endpts = arc_endpts[:, :narcs].T.copy(order="C")
    vertex_ijs = vertex_ijs[:, :nvertices].T.copy(order="C")

    if sort:
        arc_lengths = arc_endpts[:, 1] - arc_endpts[:, 0] + 1
        id = np.lexsort((arc_lengths, arc_orders))
        arc_orders = arc_orders[id]
        arc_endpts = arc_endpts[id, :]

    dsi, dsj, has_valid_ds = _valid_flow_edges(dirs, valids, dir_scheme)
    _validate_flowgraph_coverage(vertex_ijs, arc_endpts, dsi, dsj, has_valid_ds)

    if remove_unused:
        vertex_ijs, arc_endpts = remove_unused_vertices(vertex_ijs, arc_endpts)  # type: ignore

    return (
        arc_orders,
        vertex_ijs.astype(np.int32, order="C"),
        arc_endpts.astype(np.int32, order="C"),
    )
