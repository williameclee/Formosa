"""
Edits flow graphs by concatenating, splitting, and removing graph
elements.

Last modified: 2026-08-18, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np

import warnings

from numpy.typing import NDArray
from typing import Iterable, Optional, overload
from formosa.utils.typing import NpInt, NpIndex, NpCoords


def concat_flowgraph(
    orders: NDArray[NpInt],
    vtxs: NDArray[NpCoords],
    endpts: NDArray[NpIndex],
) -> tuple[NDArray[NpInt], NDArray[NpCoords], NDArray[NpIndex]]:
    """
    Concatenates arcs of the same order in a flow graph, separated
    by NaNs.
    It mainly serves to reduce the number of drawing calls when
    visualising the graph.

    Parameters
    ----------
    orders : NDArray[int]
        (O,) array representing the Strahler order for each arc in
        the flow graph.
    vtxs : NDArray[int]
        (V,2) array containing the ordered (i, j) incices of all
        arcs, concactinated together.
    ednpts : NDArray[int]
        (A,2) array containing the indices of where each arc starts
        and ends in `vtxs`.
        The returned endpoints are inclusive, meaning slicing must
        be done as `vtxs[start : end + 1]`.

    Returns
    ----------
    orders : NDArray[int]
        (O,) array representing the Strahler order for each arc in
        the flow graph.
    vtxs : NDArray[int]
        (V',2) array containing the ordered (i, j) incices of all
        arcs, concactinated together.
    ednpts : NDArray[int]
        (O,2) array containing the indices of where each arc starts
        and ends in `vtxs`.
        The returned endpoints are inclusive, meaning slicing must
        be done as `vtxs[start : end + 1]`.
    """
    # Input validation
    assert np.size(orders, 0) == np.size(endpts, 0), (
        "The order and endpoint arrays must have the same length, "
        + f"but got {np.size(orders, 0)} and {np.size(endpts, 0)}, respectively, instead"
    )
    if np.size(orders, 0) == 0:
        return orders, vtxs, endpts

    # Sort by arc order
    id = np.argsort(orders)
    orders = orders[id]
    endpts = endpts[id, :]

    s_orders, first_group_ids = np.unique(orders, return_index=True)
    arc_lengths = endpts[:, 1] - endpts[:, 0] + 1
    output_size = int(np.sum(arc_lengths) + orders.size - 1)
    output_dtype = (
        vtxs.dtype if orders.size == 1 else np.result_type(vtxs.dtype, np.float64)
    )
    s_vtxs = np.full((output_size, vtxs.shape[1]), np.nan, dtype=output_dtype)
    s_endpts = np.zeros((s_orders.size, 2), dtype=endpts.dtype)

    cursor = 0
    group_id = 0
    for iarc, (start, end) in enumerate(endpts):
        if iarc in first_group_ids:
            group_id = int(np.searchsorted(first_group_ids, iarc))
            s_endpts[group_id, 0] = cursor
        length = int(end - start + 1)
        s_vtxs[cursor : cursor + length] = vtxs[start : end + 1]
        cursor += length
        s_endpts[group_id, 1] = cursor - 1
        if iarc < orders.size - 1:
            cursor += 1

    return s_orders, s_vtxs, s_endpts


def remove_unused_vertices(
    vtxs: NDArray[NpCoords], endpts: NDArray[NpIndex]
) -> tuple[NDArray[NpCoords], NDArray[NpIndex]]:
    """
    Removes stored vertices that are not referenced by any graph
    arc.

    Arcs retain their input order and their vertices are copied into
    adjacent ranges. Consequently, the start of every arc after the
    first is one index beyond the end of the preceding arc. Arc
    endpoint indices are inclusive.

    Parameters
    ----------
    vtxs : NDArray[int | float]
        (V,n) array of stored vertex coordinates.
    endpts : NDArray[int]
        (A,2) array of inclusive arc ranges into `vtxs`.

    Returns
    -------
    vtxs : NDArray[int | float]
        Compact vertex array containing only vertices referenced by
        arcs.
    endpts : NDArray[int]
        Arc ranges remapped into the compact vertex array.

    Raises
    ------
    ValueError
        If the input arguments have the wrong shapes.
    """
    vtxs = np.asarray(vtxs)
    endpts = np.asarray(endpts)

    if vtxs.ndim != 2:
        raise ValueError("Vertices must be a two-dimensional array.")
    if endpts.ndim != 2 or endpts.shape[1] != 2:
        raise ValueError("Endpts must have shape (number of arcs, 2).")
    if endpts.shape[0] == 0:
        return vtxs[:0].copy(), endpts.copy()
    if np.any(endpts[:, 0] < 0) or np.any(endpts[:, 1] < endpts[:, 0]):
        raise ValueError(
            "Each arc must have a non-negative start no greater than its end."
        )
    if np.any(endpts[:, 1] >= vtxs.shape[0]):
        raise ValueError("Arc endpoints must index rows in vertices.")

    lengths = endpts[:, 1] - endpts[:, 0] + 1
    compact_vtxs = np.concatenate(
        [vtxs[start : end + 1] for start, end in endpts], axis=0
    )
    compact_ends = np.cumsum(lengths, dtype=np.intp) - 1
    compact_starts = np.concatenate(([0], compact_ends[:-1] + 1))
    compact_endpts = np.column_stack((compact_starts, compact_ends)).astype(
        endpts.dtype, copy=False
    )
    return compact_vtxs, compact_endpts


def _find_vertex_id(
    vtxs: NDArray[NpCoords], vtx: NDArray[NpCoords], n: Optional[int] = None
) -> int | list[int]:
    """
    Finds the index (or indices) of a vertex in a list of vertices.

    Parameters
    ----------
    verts : NDArray[int | float]
        (V,m) array representing the m-dimensional coordinates of
        the vertices.
    vtx : NDArray[int | float]
        (m,) array representing the m-dimensional coordinate of
        the vertex to find.
    n : int, optional
        Maximum number of indices to return, if the vertex appears
        multiple times in the array.
        When not specified, all occurences are returned.
        Default value is `None`.

    Returns
    -------
    ivtx : int | list[int]
        Index (or indices) of the vertex in the list of vertices.

    Raises
    ------
    AssertionError
        If the dimension of the provided vertex does not match the
        dimension of the array of vertices
    ValueError
        If the provided vertex is not found in the list of vertices.
    """

    assert np.size(vtx, 0) == np.size(vtxs, 1), (
        "The vertex and vertex array must have the same number of dimensions, "
        + f"but got {np.size(vtx, 0)} and {np.size(vtxs, 1)}, respectively, instead."
    )

    ivtx = np.squeeze(np.where(np.all(vtxs == vtx, axis=1)))
    if np.size(ivtx) == 0:
        raise ValueError("Provided vertex is not found in the list of vertices.")
    elif np.size(ivtx) > 1:
        if (n is not None) and (np.size(ivtx) > n):
            return ivtx[:n].tolist()
        return ivtx.tolist()
    return int(ivtx)


@overload
def _find_arc_id_of_vertex(
    endpts: NDArray[NpIndex], ivtx: int, is_inclusive: bool = True
) -> Optional[int]: ...


@overload
def _find_arc_id_of_vertex(
    endpts: NDArray[NpIndex], ivtx: Iterable[int], is_inclusive: bool = True
) -> Optional[list[int]]: ...


def _find_arc_id_of_vertex(
    endpts: NDArray[NpIndex], ivtx: int | Iterable[int], is_inclusive: bool = True
) -> Optional[int | list[int]]:
    """
    Finds the indices of the arcs that contain the vertices of a
    list of given indices.

    Parameters
    ----------
    endpts : NDArray[int]
        (A,2) array containing the indices of the starting and
        ending endpoint of each arc in a vertex array.
    ivtx : int | Iterable[int]
        Index or indices of the vertices in a vertex array to find
        the arcs for.
    is_inclusive : bool
        Whether the `endpts` array is inclusive or half-open.
        If it is inclusive, the corresponding vertices in the arc
        are start_id ... end_id; if half-open, the vertices are
        start_id ... end_id - 1 instead.
        Default option is `True`.

    Returns
    -------
    iarc : int | list[int], optional
        Index or indices of the arcs that contain the vertices of
        the given index or indices, or `None` if the vertices are
        not a part of any arc.
    """

    def _find_arc_of_vertex(
        endpts: NDArray[np.integer], ivtx: int, is_inclusive: bool = True
    ) -> Optional[int]:
        iarc = np.flatnonzero(
            (ivtx >= endpts[:, 0]) & (ivtx <= (endpts[:, 1] - (not is_inclusive)))
        )
        if np.size(iarc) == 0:
            warnings.warn("Provided vertex is not a part of any arc.")
            return None
        elif np.size(iarc) > 1:
            raise ValueError("Provided vertex is found in multiple arcs.")
        return iarc[0]

    if isinstance(ivtx, int) or (np.size(ivtx) == 1):  # type: ignore
        iarc = _find_arc_of_vertex(endpts, ivtx, is_inclusive)  # type: ignore
        return iarc
    iarc = [_find_arc_of_vertex(endpts, ivert, is_inclusive) for ivert in ivtx]
    iarc = [iarc_ for iarc_ in iarc if iarc_ is not None]  # Reduce the list
    return iarc


def insert_endpt(
    orders: NDArray[np.integer],
    vtxs: NDArray[NpCoords],
    endpts: NDArray[NpIndex],
    add_endpt: NDArray[NpCoords] | int,
    remove_unused: bool = False,
) -> tuple[NDArray[np.integer], NDArray[NpCoords], NDArray[NpIndex]]:
    """
    Turns an interior vertex of a flow graph in to an endpoint.

    Parameters
    ----------
    orders : NDArray[int]
        (O,) array representing the Strahler order for each arc.
    vtxs : NDArray[int | float]
        (V,n) array representing the coordinates of the vertices.
    endpts : NDArray[int]
        (A,2) array representing the indices of the starting and
        ending endpoint of each arc in the `vtxs` array.
        The endpoints should be inclusive.
    add_endpt : NDArray[int | float] | int
        Either:
        1. (n,) array representing the coordinate of the vertex to
            turn to an endpoint
        2. Integer specifying the index of the vertex in the `vtxs`
            array to turn to an endpoint
    remove_unused : bool, optional
        Whether to compact the returned vertex array so the arc
        ranges are adjacent.
        Default option is `False`.

    Returns
    -------
    orders : NDArray[int]
        Strahler order for each arc in the updated flow graph.
    vtxs : NDArray[int | float]
        Coordinates of the vertices in the updated flow graph.
    endpts : NDArray[int]
        Inclusive starting and ending vertex indices for each arc in
        the updated flow graph.

    Raises
    ------
    AssertionError
        If `orders` and `endpts` do not contain the same number of
        arcs, or if a coordinate supplied as `add_endpt` does not
        have the same dimensionality as the vertices in `vtxs`.
    """

    assert np.size(orders, 0) == np.size(endpts, 0), (
        "The orders array must have the same length as the endpoints array, "
        + f"but got {np.size(orders, 0)} and {np.size(endpts, 0)}, respectively, instead."
    )

    def _return_graph(
        orders: NDArray[np.integer],
        vtxs: NDArray[NpCoords],
        endpts: NDArray[NpIndex],
    ) -> tuple[NDArray[np.integer], NDArray[NpCoords], NDArray[NpIndex]]:
        if remove_unused:
            vtxs, endpts = remove_unused_vertices(vtxs, endpts)
        return orders, vtxs, endpts

    if isinstance(add_endpt, int):
        ivtx = add_endpt
    else:
        try:
            ivtx = _find_vertex_id(vtxs, add_endpt)
        except (AssertionError, ValueError):
            warnings.warn(
                "Provided endpoint is not found in the list of vertices. "
                + "Returning the original graph."
            )
            return _return_graph(orders, vtxs, endpts)

        # Exclude matching coordinates stored outside the ranges used by any arc
        ivtxs = np.atleast_1d(ivtx)
        useds = np.any(
            (ivtxs[:, np.newaxis] >= endpts[np.newaxis, :, 0])
            & (ivtxs[:, np.newaxis] <= endpts[np.newaxis, :, 1]),
            axis=1,
        )
        ivtxs = ivtxs[useds]
        if ivtxs.size == 0:
            warnings.warn(
                "Provided endpoint is not a part of any arc. "
                + "Returning the original graph."
            )
            return _return_graph(orders, vtxs, endpts)
        ivtx: int | list[int] = int(ivtxs[0]) if ivtxs.size == 1 else ivtxs.tolist()
    iarc = _find_arc_id_of_vertex(endpts, ivtx)

    def _insert_endpt(
        orders: NDArray[np.integer],
        ivtxs: NDArray[NpCoords],
        endpts: NDArray[NpIndex],
        iarc: int,
        ivtx: int,
    ) -> tuple[NDArray[np.integer], NDArray[NpCoords], NDArray[NpIndex]]:
        # Skip if the additional endpoint is already an endpoint
        if (endpts[iarc, 0] == ivtx) or (endpts[iarc, 1] == ivtx):
            return orders, ivtxs, endpts

        # Append the second half of the segment
        start_vert = np.size(ivtxs, 0)
        ivtxs = np.concat([ivtxs, ivtxs[ivtx : np.squeeze(endpts[iarc, 1] + 1), :]])
        end_vert = np.size(ivtxs, 0) - 1
        endpts = np.concat([endpts, np.array([[start_vert, end_vert]])])
        orders = np.concat([orders, orders[iarc : iarc + 1]])

        # Truncate the current segment to the first half
        endpts[iarc, 1] = ivtx

        return orders, ivtxs, endpts

    if isinstance(ivtx, int):
        orders, vtxs, endpts = _insert_endpt(
            orders, vtxs, endpts=endpts, iarc=iarc, ivtx=ivtx  # type: ignore
        )
        return _return_graph(orders, vtxs, endpts)

    assert isinstance(iarc, list)  # Just for static type checking
    for jvert, jarc in zip(ivtx, iarc):
        if jarc is None:
            continue
        orders, vtxs, endpts = _insert_endpt(orders, vtxs, endpts, jarc, jvert)
    return _return_graph(orders, vtxs, endpts)
