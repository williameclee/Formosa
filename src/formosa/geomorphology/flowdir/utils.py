# Last modified
#   2026-06-11, En-Chi Lee (williameclee@gmail.com)
#     - Moved auxiliary functions to this file
#     - Standardised variable and argument names
#   2026-07-01, En-Chi Lee (williameclee@gmail.com)
#     - Made out-of-bound check in `compute_downstream_indices` optional
#   2026-07-02, En-Chi Lee (williameclee@gmail.com)
#     - Actually implemented validity check in `compute_downstram_indices`
#     - Updated `geomorphology.flowdir` submodule path

import numpy as np

from formosa.geomorphology.flowdir.d8directions import D8Directions

import warnings

import numpy.typing as npt
from typing import Optional


def get_neighbour_values(
    array: np.ndarray,
    dir_scheme: D8Directions = D8Directions(),
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
    dir_scheme : D8Directions, optional
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
    pad_width = np.max(abs(dir_scheme.offsets))
    array_padded = np.pad(
        array,
        pad_width=pad_width,
        mode="constant",
        constant_values=pad_value,
    )
    neighbours = np.zeros((len(dir_scheme.codes), *array.shape), dtype=array.dtype)
    offsets = np.zeros((len(dir_scheme.codes), 2), dtype=np.int16)
    for i_offset, [di, dj] in enumerate(dir_scheme.offsets.astype(np.int16)):
        offsets[i_offset, :] = [di, dj]
        neighbours[i_offset, :, :] = array_padded[
            pad_width + di : pad_width + di + array.shape[0],
            pad_width + dj : pad_width + dj + array.shape[1],
        ]

    codes = dir_scheme.codes
    if not include_self:
        # exclude self (first offset)
        self_id = np.where(np.all(dir_scheme.offsets == [0, 0], axis=1))[0][0]
        neighbours = np.delete(neighbours, self_id, axis=0)
        codes = np.delete(codes, self_id, axis=0)
        offsets = np.delete(offsets, self_id, axis=0)
    elif self_at_last:
        neighbours = np.roll(neighbours, -1, axis=0)
        codes = np.roll(codes, -1, axis=0)
        offsets = np.roll(offsets, -1, axis=0)
    return neighbours, codes, offsets


def compute_downstream_indices(
    dirs: npt.NDArray[np.integer],
    dir_scheme: D8Directions = D8Directions(),
    valids: Optional[npt.NDArray[np.bool_]] = None,
    check: bool = True,
) -> tuple[
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    npt.NDArray[np.bool_] | None,
]:
    """
    Computes the downstream indices for each cell in a flow direction grid.

    Parameters
    ----------
    dirs : NDArray[int]
        A 2D array representing the flow directions for each cell.
    dir_scheme : D8Directions, optional
        An instance of D8Directions defining the flow direction scheme.
        Default is D8Directions().
    valids : NDArray[bool], optional
        A boolean mask array indicating valid cells in the flow direction grid.
        If None, all cells are considered valid.
        Default is None.
    check : bool, optional
        Whether to raise an error if some downstream indices are out of bounds.
        Otherwise, only a warning is issued.
        Default is True.

    Returns
    -------
    dsi : NDArray[int]
        A 2D array of downstream row indices for each cell.
        When the cell is invalid, it is set to -1.
    dsj : NDArray[int]
        A 2D array of downstream column indices for each cell.
        When the cell is invalid, it is set to -1.
    dsij : NDArray[int32]
        A 2D array of flattened downstream indices for each cell.
        When the cell is invalid, it is set to -1.
    ds_inbounds : NDArray[bool]
        A boolean mask array indicating out-of-bound downstream cells for each cell.

    Raises
    ------
    ValueError
        If `check` is `True` and some downstream indices are out of bounds.
    UserWarning
        If `check` is `False` but some downstream indices are out of bounds.
    """
    if valids is None:
        valids = ~np.isnan(dirs)
    elif isinstance(valids, np.ndarray):
        assert (
            valids.shape == dirs.shape
        ), f"Shapes for flow direction ({dirs.shape}) and valid mask ({valids.shape}) do not match."
    else:
        raise TypeError(
            f"Expected valids to be None or np.ndarray, got {type(valids)} instead."
        )

    I, J = dirs.shape
    ii, jj = np.meshgrid(
        np.arange(I, dtype=np.int32), np.arange(J, dtype=np.int32), indexing="ij"
    )
    di, dj = dir_scheme.code2d8offset(dirs)
    dsi = ii.astype(np.int32) + (di).astype(np.int32)
    dsj = jj.astype(np.int32) + (dj).astype(np.int32)
    dsij: npt.NDArray[np.int32] = dsj.astype(np.int32) * I + dsi.astype(np.int32)

    dsi[~valids] = -1
    dsj[~valids] = -1
    dsij[~valids] = -1

    ds_oobs = valids & ((dsi < 0) | (dsi >= I) | (dsj < 0) | (dsj >= J))

    if not np.any(ds_oobs):
        return dsi, dsj, dsij, np.full(dirs.shape, True, dtype=bool)

    if check:
        raise ValueError("Some downstream indices out of bounds")

    warnings.warn("Some downstream indices out of bounds", UserWarning)
    return dsi, dsj, dsij, ~ds_oobs
