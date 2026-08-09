"""
Edits flow graphs by concatenating, splitting, and removing graph
elements.

Last modified: 2026-08-10, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np

import warnings

import numpy.typing as npt
from typing import Iterable, Optional, TypeVar, overload

NpIndex = TypeVar("NpIndex", np.int32, np.int64, np.intp)


def concat_flowgraph(
    arc_orders: npt.NDArray[np.integer],
    vertex_ijs: npt.NDArray[np.integer],
    arc_endpts: npt.NDArray[np.integer],
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer], npt.NDArray[np.integer]]:
    """
    Concatenates arcs of the same order in a flow graph, separated by NaNs.
    It mainly serves to reduce the number of drawing calls when visualising the graph.

    Parameters
    ----------
    arc_orders : NDArray[int]
        O-by-1 array representing the Strahler order for each arc in the flow graph.
    vertex_ijs : NDArray[int]
        V-by-2 array containing the ordered (i, j) incices of all arcs, concactinated together.
    vertex_startends : NDArray[int]
        A-by-2 array containing the indices of where each arc starts and ends in `vertex_ijs`.
        The returned endpoints are inclusive, meaning slicing must be done as `vertex_ijs[start : end + 1]`.

    Returns
    ----------
    arc_orders : NDArray[int]
        O-by-1 array representing the Strahler order for each arc in the flow graph.
    vertex_ijs : NDArray[int]
        V'-by-2 array containing the ordered (i, j) incices of all arcs, concactinated together.
    vertex_startends : NDArray[int]
        O-by-2 array containing the indices of where each arc starts and ends in `vertex_ijs`.
        The returned endpoints are inclusive, meaning slicing must be done as `vertex_ijs[start : end + 1]`.
    """
    # Input validation
    assert np.size(arc_orders, 0) == np.size(arc_endpts, 0), (
        "The order and endpoint arrays must have the same length, "
        + f"but got {np.size(arc_orders, 0)} and {np.size(arc_endpts, 0)}, respectively, instead"
    )
    if np.size(arc_orders, 0) == 0:
        return arc_orders, vertex_ijs, arc_endpts

    # Sort by arc order
    id = np.argsort(arc_orders)
    arc_orders = arc_orders[id]
    arc_endpts = arc_endpts[id, :]

    s_arc_orders, first_group_ids = np.unique(arc_orders, return_index=True)
    arc_lengths = arc_endpts[:, 1] - arc_endpts[:, 0] + 1
    output_size = int(np.sum(arc_lengths) + arc_orders.size - 1)
    output_dtype = (
        vertex_ijs.dtype
        if arc_orders.size == 1
        else np.result_type(vertex_ijs.dtype, np.float64)
    )
    s_vertex_ijs = np.full(
        (output_size, vertex_ijs.shape[1]), np.nan, dtype=output_dtype
    )
    s_arc_endpts = np.zeros((s_arc_orders.size, 2), dtype=np.int32)

    cursor = 0
    group_id = 0
    for iarc, (start, end) in enumerate(arc_endpts):
        if iarc in first_group_ids:
            group_id = int(np.searchsorted(first_group_ids, iarc))
            s_arc_endpts[group_id, 0] = cursor
        length = int(end - start + 1)
        s_vertex_ijs[cursor : cursor + length] = vertex_ijs[start : end + 1]
        cursor += length
        s_arc_endpts[group_id, 1] = cursor - 1
        if iarc < arc_orders.size - 1:
            cursor += 1

    return s_arc_orders, s_vertex_ijs, s_arc_endpts


def remove_unused_vertices(
    vertices: npt.NDArray[np.number],
    endpts: npt.NDArray[NpIndex],
) -> tuple[npt.NDArray[np.number], npt.NDArray[NpIndex]]:
    """
    Removes stored vertices that are not referenced by any graph arc.

    Arcs retain their input order and their vertices are copied into adjacent
    ranges. Consequently, the start of every arc after the first is one index
    beyond the end of the preceding arc. Arc endpoint indices are inclusive.

    Parameters
    ----------
    vertices : NDArray[int | float]
        V-by-n array of stored vertex coordinates.
    endpts : NDArray[int]
        A-by-2 array of inclusive arc ranges into `vertices`.

    Returns
    -------
    vertices : NDArray[int | float]
        Compact vertex array containing only vertices referenced by arcs.
    endpts : NDArray[int]
        Arc ranges remapped into the compact vertex array.

    Raises
    ------
    ValueError
        If the input arguments have the wrong shapes.
    """
    vertices = np.asarray(vertices)
    endpts = np.asarray(endpts)

    if vertices.ndim != 2:
        raise ValueError("vertices must be a two-dimensional array.")
    if endpts.ndim != 2 or endpts.shape[1] != 2:
        raise ValueError("endpts must have shape (number of arcs, 2).")
    if endpts.shape[0] == 0:
        return vertices[:0].copy(), endpts.copy()
    if np.any(endpts[:, 0] < 0) or np.any(endpts[:, 1] < endpts[:, 0]):
        raise ValueError(
            "Each arc must have a non-negative start no greater than its end."
        )
    if np.any(endpts[:, 1] >= vertices.shape[0]):
        raise ValueError("Arc endpoints must index rows in vertices.")

    lengths = endpts[:, 1] - endpts[:, 0] + 1
    compact_vertices = np.concatenate(
        [vertices[start : end + 1] for start, end in endpts], axis=0
    )
    compact_ends = np.cumsum(lengths, dtype=np.intp) - 1
    compact_starts = np.concatenate(([0], compact_ends[:-1] + 1))
    compact_endpts = np.column_stack((compact_starts, compact_ends)).astype(
        endpts.dtype, copy=False
    )
    return compact_vertices, compact_endpts


def _find_vertex_id(
    verts: npt.NDArray[np.number],
    vert: npt.NDArray[np.number],
    n: Optional[int] = None,
) -> int | list[int]:
    """
    Finds the index (or indices) of a vertex in a list of vertices.

    Parameters
    ----------
    verts : NDArray[int | float]
        V-by-m array representing the m-dimensional coordinates of the vertices.
    vert : NDArray[int | float]
        m-by-(1) array representing the m-dimensional coordinate of the vertex to find.
    n : int, optional
        Maximum number of indices to return, if the vertex appears multiple times in the array.
        When not specified, all occurences are returned.
        Default value is `None`.

    Returns
    -------
    ivert : int | list[int]
        Index (or indices) of the vertex in the list of vertices

    Raises
    ------
    AssertionError
        If the dimension of the provided vertex does not match the dimension of the array of vertices
    ValueError
        If the provided vertex is not found in the list of vertices
    """

    assert np.size(vert, 0) == np.size(verts, 1), (
        "The vertex and vertex array must have the same number of dimensions, "
        + f"but got {np.size(vert, 0)} and {np.size(verts, 1)}, respectively, instead."
    )

    ivert = np.squeeze(np.where(np.all(verts == vert, axis=1)))
    if np.size(ivert) == 0:
        raise ValueError("Provided vertex is not found in the list of vertices.")
    elif np.size(ivert) > 1:
        if (n is not None) and (np.size(ivert) > n):
            return ivert[:n].tolist()
        return ivert.tolist()
    return int(ivert)


@overload
def _find_arc_id_of_vertex(
    endpts: npt.NDArray[np.integer], ivert: int, is_inclusive: bool = True
) -> Optional[int]: ...


@overload
def _find_arc_id_of_vertex(
    endpts: npt.NDArray[np.integer], ivert: Iterable[int], is_inclusive: bool = True
) -> Optional[list[int]]: ...


def _find_arc_id_of_vertex(
    endpts: npt.NDArray[np.integer],
    ivert: int | Iterable[int],
    is_inclusive: bool = True,
) -> Optional[int | list[int]]:
    """
    Finds the indices of the arcs that contain the vertices of a list of given indices.

    Parameters
    ----------
    endpts : NDArray[int]
        A-by-2 array containing the indices of the starting and ending endpoint of each arc in a vertex array.
    ivert : int | Iterable[int]
        Index or indices of the vertices in a vertex array to find the arcs for.
    is_inclusive : bool
        Whether the `endpts` array is inclusive or half-open.
        If it is inclusive, the corresponding vertices in the arc are start_id ... end_id; if half-open, the vertices are start_id ... end_id - 1 instead.
        Default option is `True`.

    Returns
    -------
    iarc : int | list[int], optional
        Index or indices of the arcs that contain the vertices of the given index or indices, or `None` if the vertices are not a part of any arc.
    """

    def _find_arc_of_vertex(
        endpts: npt.NDArray[np.integer], ivert: int, is_inclusive: bool = True
    ) -> Optional[int]:
        iarc = np.flatnonzero(
            (ivert >= endpts[:, 0]) & (ivert <= (endpts[:, 1] - (not is_inclusive)))
        )
        if np.size(iarc) == 0:
            warnings.warn("Provided vertex is not a part of any arc.")
            return None
        elif np.size(iarc) > 1:
            raise ValueError("Provided vertex is found in multiple arcs.")
        return iarc[0]

    if isinstance(ivert, int) or (np.size(ivert) == 1):  # type: ignore
        iarc = _find_arc_of_vertex(endpts, ivert, is_inclusive)  # type: ignore
        return iarc
    iarc = [_find_arc_of_vertex(endpts, ivert, is_inclusive) for ivert in ivert]
    iarc = [iarc_ for iarc_ in iarc if iarc_ is not None]  # Reduce the list
    return iarc


def insert_endpt(
    orders: npt.NDArray[np.integer],
    ijs: npt.NDArray[np.number],
    endpts: npt.NDArray[np.integer],
    add_endpt: npt.NDArray[np.number] | int,
    remove_unused: bool = False,
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.number], npt.NDArray[np.integer]]:
    """
    Turns an interior vertex of a flow graph in to an endpoint.

    Parameters
    ----------
    orders : NDArray[int]
        O-by-(1) array representing the Strahler order for each arc.
    ijs : NDArray[int | float]
        V-by-n array representing the coordinates of the vertices.
    endpts : NDArray[int]
        A-by-2 array representing the indices of the starting and ending endpoint of each arc in the `ijs` array.
        The endpoints should be inclusive.
    add_endpt : NDArray[int | float] | int
        Either:
         1. n-by-(1) array representing the coordinate of the vertex to turn to an endpoint
         2. Integer specifying the index of the vertex in the `ijs` array to turn to an endpoint
    remove_unused : bool, optional
        Whether to compact the returned vertex array so the arc ranges are adjacent.
        Default option is `False`.

    Returns
    -------
    orders : NDArray[int]
        Strahler order for each arc in the updated flow graph.
    ijs : NDArray[int | float]
        Coordinates of the vertices in the updated flow graph.
    endpts : NDArray[int]
        Inclusive starting and ending vertex indices for each arc in the updated flow graph.

    Raises
    ------
    AssertionError
        If `orders` and `endpts` do not contain the same number of arcs, or if a
        coordinate supplied as `add_endpt` does not have the same dimensionality
        as the vertices in `ijs`.
    """

    assert np.size(orders, 0) == np.size(endpts, 0), (
        "The orders array must have the same length as the endpoints array, "
        + f"but got {np.size(orders, 0)} and {np.size(endpts, 0)}, respectively, instead."
    )

    def _return_graph(orders, ijs, endpts):
        if remove_unused:
            ijs, endpts = remove_unused_vertices(ijs, endpts)
        return orders, ijs, endpts

    if isinstance(add_endpt, int):
        ivert = add_endpt
    else:
        try:
            ivert = _find_vertex_id(ijs, add_endpt)
        except (AssertionError, ValueError):
            warnings.warn(
                "Provided endpoint is not found in the list of vertices. "
                + "Returning the original graph."
            )
            return _return_graph(orders, ijs, endpts)

        # Exclude matching coordinates stored outside the ranges used by any arc
        iverts = np.atleast_1d(ivert)
        useds = np.any(
            (iverts[:, np.newaxis] >= endpts[np.newaxis, :, 0])
            & (iverts[:, np.newaxis] <= endpts[np.newaxis, :, 1]),
            axis=1,
        )
        iverts = iverts[useds]
        if iverts.size == 0:
            warnings.warn(
                "Provided endpoint is not a part of any arc. "
                + "Returning the original graph."
            )
            return _return_graph(orders, ijs, endpts)
        ivert: int | list[int] = int(iverts[0]) if iverts.size == 1 else iverts.tolist()
    iarc = _find_arc_id_of_vertex(endpts, ivert)

    def _insert_endpt(orders, ijs, endpts, iarc, ivert):
        # Skip if the additional endpoint is already an endpoint
        if (endpts[iarc, 0] == ivert) or (endpts[iarc, 1] == ivert):
            return orders, ijs, endpts

        # Append the second half of the segment
        start_vert = np.size(ijs, 0)
        ijs = np.concat([ijs, ijs[ivert : np.squeeze(endpts[iarc, 1] + 1), :]])
        end_vert = np.size(ijs, 0) - 1
        endpts = np.concat([endpts, np.array([[start_vert, end_vert]])])
        orders = np.concat([orders, orders[iarc : iarc + 1]])

        # Truncate the current segment to the first half
        endpts[iarc, 1] = ivert

        return orders, ijs, endpts

    if isinstance(ivert, int):
        orders, ijs, endpts = _insert_endpt(orders, ijs, endpts, iarc, ivert)
        return _return_graph(orders, ijs, endpts)

    assert isinstance(iarc, list)  # Just for static type checking
    for jvert, jarc in zip(ivert, iarc):
        if jarc is None:
            continue
        orders, ijs, endpts = _insert_endpt(orders, ijs, endpts, jarc, jvert)
    return _return_graph(orders, ijs, endpts)
