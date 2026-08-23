"""
Locates neighbouring raster cells and retrieves their values.

Created: 2026-08-01, En-Chi Lee (williameclee@gmail.com)
Last modified: 2026-08-23, En-Chi Lee (williameclee@gmail.com)
"""

import warnings

import numpy as np
from numpy.typing import NDArray

from formosa.geomorphology.drainage.directions import D8Directions
from formosa.geomorphology.raster_validation import (
    validate_format_flowdirs,
    validate_format_valids,
)
from formosa.utils import NpCanonIndex, NpFlowDir


def get_neighbour_values(
    array: np.ndarray,
    dir_scheme: D8Directions = D8Directions(),
    pad_val: np.number | float = np.nan,
    include_self: bool = False,
    self_at_last: bool = False,
) -> tuple[np.ndarray, NDArray[np.integer], NDArray[np.integer]]:
    """
    Gets the values of neighbouring cells in an array based on specified directions.

    Parameters
    ----------
    array : NDArray
        Raster array from which to extract neighbour values.
        - Expected shape: `(nrows, ncols)`.
    dir_scheme : D8Directions, optional
        Instance of `D8Directions` defining the neighbour offsets.
        - Default scheme is `D8Directions()`.
    pad_val : int | float, optional
        Value to use for padding the array edges.
        - Default value is `np.nan`.
    include_self : bool, optional
        Whether to include the value of the cell itself as a
        neighbour.
        - Default option is `False`.
    self_at_last : bool, optional
        If `include_self` is True, whether to place the self value
        at the end of the neighbour list.
        - Default option is `False`.

    Returns
    -------
    nabrs : NDArray
        Extracted neighbour values.
        - Shape: `(N, nrows, ncols)`.
    codes : NDArray[int]
        Direction codes corresponding to the neighbours.
        - Shape: `(N,)`.
    offsets : NDArray[int]
        Row and column offsets (di, dj) corresponding to the
        neighbours.
        - Shape: `(N, 2)`.
    """
    # Input validation and initialisation
    if np.issubdtype(array.dtype, np.integer) and pad_val is np.nan:
        warnings.warn(
            "Integer array does not support NaN padding, using max int instead"
        )
        pad_val = np.iinfo(array.dtype).max

    # Main
    # get padding width from offset
    pad_width = np.max(abs(dir_scheme.offsets))
    array_padded = np.pad(
        array,
        pad_width=pad_width,
        mode="constant",
        constant_values=pad_val,
    )
    nabrs = np.zeros((len(dir_scheme.codes), *array.shape), dtype=array.dtype)
    offsets = np.zeros((len(dir_scheme.codes), 2), dtype=np.int16)
    for i_offset, [di, dj] in enumerate(dir_scheme.offsets.astype(np.int16)):
        offsets[i_offset, :] = [di, dj]
        nabrs[i_offset, :, :] = array_padded[
            pad_width + di : pad_width + di + array.shape[0],
            pad_width + dj : pad_width + dj + array.shape[1],
        ]

    codes = dir_scheme.codes
    if not include_self:
        # exclude self (first offset)
        self_id = np.where(np.all(dir_scheme.offsets == [0, 0], axis=1))[0][0]
        nabrs = np.delete(nabrs, self_id, axis=0)
        codes = np.delete(codes, self_id, axis=0)
        offsets = np.delete(offsets, self_id, axis=0)
    elif self_at_last:
        nabrs = np.roll(nabrs, -1, axis=0)
        codes = np.roll(codes, -1, axis=0)
        offsets = np.roll(offsets, -1, axis=0)
    return nabrs, codes, offsets


def compute_downstream_indices(
    dirs: NDArray[NpFlowDir],
    dir_scheme: D8Directions = D8Directions(),
    valids: NDArray[np.bool_] | None = None,
    check: bool = True,
    return_flat_index: bool = True,
    oob_is_okay: bool = False,
) -> tuple[
    NDArray[NpCanonIndex],
    NDArray[NpCanonIndex],
    NDArray[NpCanonIndex] | None,
    NDArray[np.bool_],
]:
    """
    Computes the downstream indices for each cell in a flow direction grid.

    Parameters
    ----------
    dirs : NDArray[uint8]
        Flow direction raster.
        - Expected shape: `(nrows, ncols)`.
    dir_scheme : D8Directions, optional
        Instance of `D8Directions` defining the flow direction
        scheme.
        - Default scheme is `D8Directions()`.
    valids : NDArray[bool], optional
        Boolean mask indicating valid cells in the flow direction
        grid.
        If `None`, all cells are considered valid.
        - Expected shape: `(nrows, ncols)`, same as `dirs`.
        - Default mask is `None`.
    check : bool, optional
        Whether to raise an error if some downstream indices are out of bounds.
        Otherwise, only a warning is issued.
        Default option is `True`.
    return_flat_index : bool, optional
        Whether to compute the flattened downstream indices.
        Defualt option is `True`.
    oob_is_okay: bool, optional
        Whether having out-of-bound downstream cells is expected (or will be explicitly handled downstream).
        Default behaviour is `False`.

    Returns
    -------
    dsi : NDArray[int32]
        Downstream row indices for each cell.
        When the cell is invalid, it is set to `-1`.
        - Shape: `(nrows, ncols)`, same as `dirs`.
    dsj : NDArray[int32]
        Downstream column indices for each cell.
        When the cell is invalid, it is set to `-1`.
        - Shape: `(nrows, ncols)`, same as `dirs`.
    dsij : NDArray[int32] | None
        Flattened downstream indices for each cell when
        `return_flat_index` is `True`, or `None` otherwise.
        When the cell is invalid, it is set to `-1`.
        - Shape: `(nrows, ncols)`, same as `dirs` (when present).
    ds_inbounds : NDArray[bool]
        Boolean mask indicating in-bounds downstream cells for each
        cell.
        - Shape: `(nrows, ncols)`, same as `dirs`.

    Warns
    -----
    UserWarning
        If `check` is `False` and `oob_is_okay` is `False`, but some
        downstream indices are out of bounds.
    """
    dirs = validate_format_flowdirs(dirs)
    valids = validate_format_valids(valids, dirs, "flow direction raster")

    I, J = dirs.shape
    ii, jj = np.meshgrid(
        np.arange(I, dtype=np.int32), np.arange(J, dtype=np.int32), indexing="ij"
    )
    di, dj = dir_scheme.code2d8offset(dirs)
    dsi = ii.astype(np.int32) + (di).astype(np.int32)
    dsj = jj.astype(np.int32) + (dj).astype(np.int32)

    dsi[~valids] = -1
    dsj[~valids] = -1

    if return_flat_index:
        dsij = dsj.astype(np.int32) * I + dsi.astype(np.int32)
        dsij[~valids] = -1
        dsij = dsij.astype(NpCanonIndex)
    else:
        dsij = None

    ds_oobs = valids & ((dsi < 0) | (dsi >= I) | (dsj < 0) | (dsj >= J))

    if not np.any(ds_oobs):
        return dsi, dsj, dsij, np.full(dirs.shape, True, dtype=bool)

    if check:
        raise ValueError("Some downstream indices out of bounds")

    if not oob_is_okay:
        warnings.warn("Some downstream indices out of bounds", UserWarning)
    return dsi.astype(NpCanonIndex), dsj.astype(NpCanonIndex), dsij, ~ds_oobs
