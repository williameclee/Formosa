# Last modified
#   2026-06-11, En-Chi Lee (williameclee@gmail.com)
#     - Moved auxiliary functions to this file

import numpy as np

from formosa.geomorphology.d8directions import D8Directions

import numpy.typing as npt
from typing import Optional


def get_neighbour_values(
    array: np.ndarray,
    directions: D8Directions = D8Directions(),
    pad_value: np.number | float | int = np.nan,
    include_self: bool = False,
    self_at_last: bool = False,
) -> tuple[np.ndarray, npt.NDArray[np.integer], npt.NDArray[np.integer]]:
    """
    Gets the values of neighbouring cells in an array based on specified directions.

    Parameters
    ----------
    array : NDArray
        A 2D array from which to extract neighbour values.
    directions : D8Directions, optional
        An instance of D8Directions defining the neighbour offsets.
        Default is D8Directions().
    pad_value : number | float | int, optional
        Value to use for padding the array edges (default is np.nan).
    include_self : bool, optional
        Whether to include the value of the cell itself as a neighbour (default is False).
    self_at_last : bool, optional
        If include_self is True, whether to place the self value at the end of the neighbour list (default is False).

    Returns
    -------
    neighbours : NDArray
        A 3D array where the first dimension corresponds to neighbour indices and the other two dimensions match the input array.
    codes : NDArray[int]
        A 1D array of direction codes corresponding to the neighbours.
    offsets : NDArray[int]
        A 2D array of offsets (di, dj) corresponding to the neighbours.
    """
    # Input validation and initialisation
    if np.issubdtype(array.dtype, np.integer) and pad_value is np.nan:
        Warning("Integer array does not support NaN padding, using max int instead")
        pad_value = np.iinfo(array.dtype).max

    # Main
    # get padding width from offset
    pad_width = np.max(abs(directions.offsets))
    array_padded = np.pad(
        array,
        pad_width=pad_width,
        mode="constant",
        constant_values=pad_value,
    )
    neighbours = np.zeros((len(directions.codes), *array.shape), dtype=array.dtype)
    offsets = np.zeros((len(directions.codes), 2), dtype=np.int16)
    for i_offset, [di, dj] in enumerate(directions.offsets.astype(np.int16)):
        offsets[i_offset, :] = [di, dj]
        neighbours[i_offset, :, :] = array_padded[
            pad_width + di : pad_width + di + array.shape[0],
            pad_width + dj : pad_width + dj + array.shape[1],
        ]

    codes = directions.codes
    if not include_self:
        # exclude self (first offset)
        self_id = np.where(np.all(directions.offsets == [0, 0], axis=1))[0][0]
        neighbours = np.delete(neighbours, self_id, axis=0)
        codes = np.delete(codes, self_id, axis=0)
        offsets = np.delete(offsets, self_id, axis=0)
    elif self_at_last:
        neighbours = np.roll(neighbours, -1, axis=0)
        codes = np.roll(codes, -1, axis=0)
        offsets = np.roll(offsets, -1, axis=0)
    return neighbours, codes, offsets


def compute_downstream_indices(
    flowdirs: npt.NDArray[np.integer],
    directions: D8Directions = D8Directions(),
    valids: Optional[npt.NDArray[np.bool_]] = None,
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer], npt.NDArray[np.int32]]:
    """
    Computes the downstream indices for each cell in a flow direction grid.

    Parameters
    ----------
    flowdirs : NDArray[int]
        A 2D array representing the flow directions for each cell.
    directions : D8Directions, optional
        An instance of D8Directions defining the flow direction scheme.
        Default is D8Directions().
    valids : NDArray[bool], optional
        A boolean mask array indicating valid cells in the flow direction grid.
        If None, all cells are considered valid.
        Default is None.

    Returns
    -------
    dsi : NDArray[int]
        A 2D array of downstream row indices for each cell.
    dsj : NDArray[int]
        A 2D array of downstream column indices for each cell.
    dsij : NDArray[int32]
        A 2D array of flattened downstream indices for each cell.
    """
    if valids is None:
        valids = ~np.isnan(flowdirs)
    elif isinstance(valids, np.ndarray):
        assert (
            valids.shape == flowdirs.shape
        ), f"Shapes for flow direction ({flowdirs.shape}) and valid mask ({valids.shape}) do not match."
    else:
        raise TypeError(
            f"Expected valids to be None or np.ndarray, got {type(valids)} instead."
        )

    I, J = flowdirs.shape
    ii, jj = np.meshgrid(
        np.arange(I, dtype=np.int32), np.arange(J, dtype=np.int32), indexing="ij"
    )
    di, dj = directions.code2d8offset(flowdirs)
    dsi = (ii.astype(np.int16) + (di).astype(np.int16)).astype(np.int16)
    dsj = (jj.astype(np.int16) + (dj).astype(np.int16)).astype(np.int16)
    dsij: npt.NDArray[np.int32] = dsj.astype(np.int32) * I + dsi.astype(np.int32)

    if np.any((dsi < 0) | (dsi >= I) | (dsj < 0) | (dsj >= J)):
        raise ValueError("Some downstream indices out of bounds")

    return dsi, dsj, dsij
