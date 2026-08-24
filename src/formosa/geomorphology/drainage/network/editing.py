"""
Edits flow graphs by concatenating, splitting, and removing graph
elements.

Last modified: 2026-08-23, En-Chi Lee (williameclee@gmail.com)
"""

import warnings
from collections.abc import Iterable
from typing import overload

import numpy as np
from numpy.typing import NDArray

from formosa.utils.typing import NpCoords, NpIndex, NpInt


def concat_flowgraph[O: NpInt, V: NpCoords, E: NpIndex](
    orders: NDArray[O], vtxs: NDArray[V], endpts: NDArray[E]
) -> tuple[NDArray[O], NDArray[V], NDArray[E]]:
    """
    Concatenates arcs of the same order in a flow graph, separated
    by NaNs.

    This mainly serves to reduce the number of drawing calls when
    visualising the graph.

    Parameters
    ----------
    orders : NDArray[int]
        Strahler order for each arc in the flow graph.
        - Expected shape: `(narcs,)`.
    vtxs : NDArray[int | float]
        Ordered (i, j) indices of all arcs concatenated together.
        - Expected shape: `(nvtxs, 2)`.
    endpts : NDArray[int]
        Indices of where each arc starts and ends in `vtxs`.
        The returned endpoints are inclusive, meaning slicing must
        be done as `vtxs[start : end + 1]`.
        - Expected shape: `(narcs, 2)`.

    Returns
    -------
    orders : NDArray[int]
        Unique arc Strahler orders.
        - Shape: `(norders,)`.
    vtxs : NDArray[int | float]
        Ordered (i, j) indices of all arcs concatenated together.
        - Shape: `(nvtxs_out, 2)`.
    endpts : NDArray[int]
        Indices of where each concatenated arc starts and ends in `vtxs`.
        - Shape: `(norders, 2)`.
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


def remove_unused_vertices[V: NpCoords, E: NpIndex](
    vtxs: NDArray[V], endpts: NDArray[E]
) -> tuple[NDArray[V], NDArray[E]]:
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
    """
    vtxs = np.asarray(vtxs)
    endpts = np.asarray(endpts)

    if vtxs.ndim != 2:
        raise ValueError("Vertices must be a 2D array.")
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


def _find_vertex_id[V: NpCoords](
    vtxs: NDArray[V], vtx: NDArray[V], n: int | None = None
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
def find_arc_id_of_vertex(
    endpts: NDArray[NpIndex], ivtx: int, inclusive: bool = True
) -> int | None: ...


@overload
def find_arc_id_of_vertex(
    endpts: NDArray[NpIndex], ivtx: Iterable[int], inclusive: bool = True
) -> list[int | None]: ...


def find_arc_id_of_vertex(
    endpts: NDArray[NpIndex], ivtx: int | Iterable[int], inclusive: bool = True
) -> int | None | list[int | None]:
    """
    Finds the indices of the arcs that contain the vertices of a
    list of given indices.

    Parameters
    ----------
    endpts : NDArray[int]
        Indices of the starting and ending endpoint of each arc in
        a vertex array.
        - Expected shape: `(narcs, 2)`.
    ivtx : int | Iterable[int]
        Index or indices of the vertices in a vertex array to find
        the arcs for.
    inclusive : bool, optional
        Whether the `endpts` array is inclusive or half-open.
        If it is inclusive, the corresponding vertices in the arc
        are start_id ... end_id; if half-open, the vertices are
        start_id ... end_id - 1 instead.
        - Default option is `True`.

    Returns
    -------
    iarc : int | None | list[int | None]
        Index or indices of the arcs that contain the vertices of
        the given index or indices, or `None` if the vertices are
        not a part of any arc.
    """

    def _find_arc_of_vertex(
        endpts: NDArray[np.integer], ivtx: int, is_inclusive: bool = True
    ) -> int | None:
        iarc = np.flatnonzero(
            (ivtx >= endpts[:, 0]) & (ivtx <= (endpts[:, 1] - (not is_inclusive)))
        )
        if np.size(iarc) == 0:
            warnings.warn("Provided vertex is not a part of any arc.")
            return None
        elif np.size(iarc) > 1:
            raise ValueError("Provided vertex is found in multiple arcs.")
        return iarc[0]

    if isinstance(ivtx, int):
        iarc = _find_arc_of_vertex(endpts, ivtx, inclusive)
        return iarc
    iarc = [_find_arc_of_vertex(endpts, ivert, inclusive) for ivert in ivtx]
    return iarc


def insert_endpt[O: NpInt, V: NpCoords, E: NpIndex](
    orders: NDArray[O],
    vtxs: NDArray[V],
    endpts: NDArray[E],
    add_endpt: NDArray[V] | int,
    remove_unused: bool = False,
) -> tuple[NDArray[O], NDArray[V], NDArray[E]]:
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
    add_endpt : NDArray[float] | int
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
    """

    assert np.size(orders, 0) == np.size(endpts, 0), (
        "The orders array must have the same length as the endpoints array, "
        + f"but got {np.size(orders, 0)} and {np.size(endpts, 0)}, respectively, instead."
    )

    def _return_graph(
        orders: NDArray[O], vtxs: NDArray[V], endpts: NDArray[E]
    ) -> tuple[NDArray[O], NDArray[V], NDArray[E]]:
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
    jarc = find_arc_id_of_vertex(endpts, ivtx)

    def _insert_endpt(
        orders: NDArray[O], ivtxs: NDArray[V], endpts: NDArray[E], iarc: int, ivtx: int
    ) -> tuple[NDArray[O], NDArray[V], NDArray[E]]:
        # Skip if the additional endpoint is already an endpoint
        if (endpts[iarc, 0] == ivtx) or (endpts[iarc, 1] == ivtx):
            return orders, ivtxs, endpts

        # Append the second half of the segment
        start_vert = np.size(ivtxs, 0)
        ivtxs = np.concat(
            [
                ivtxs,
                ivtxs[ivtx : np.squeeze(endpts[iarc, 1] + 1), :],
            ]
        )
        end_vert = np.size(ivtxs, 0) - 1
        endpts = np.concat([endpts, np.array([[start_vert, end_vert]])])
        orders = np.concat([orders, orders[iarc : iarc + 1]])

        # Truncate the current segment to the first half
        endpts[iarc, 1] = ivtx

        return orders, ivtxs, endpts

    if isinstance(ivtx, int):
        jarc = find_arc_id_of_vertex(endpts, ivtx)
        if jarc is None:
            return _return_graph(orders, vtxs, endpts)
        orders, vtxs, endpts = _insert_endpt(
            orders,
            vtxs,
            endpts=endpts,
            iarc=jarc,
            ivtx=ivtx,  # type: ignore
        )
        return _return_graph(orders, vtxs, endpts)

    iarcs = find_arc_id_of_vertex(endpts, ivtx)
    for jvtx, jarc in zip(ivtx, iarcs):
        if jarc is None:
            continue
        orders, vtxs, endpts = _insert_endpt(orders, vtxs, endpts, jarc, jvtx)
    return _return_graph(orders, vtxs, endpts)
