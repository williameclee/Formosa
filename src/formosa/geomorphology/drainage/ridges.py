"""
Derives and analyses ridge networks from digital elevation models.

The method treats the maximum confluence distance between a raster
cell and its neighbours as a proxy for ridge likelihood, then
applies conventional drainage network operations to the reciprocal
field.

Created: 2026-08-01, En-Chi Lee (williameclee@gmail.com)
Last modified: 2026-08-23, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np
from numpy.typing import NDArray

import formosa.geomorphology.drainage.flowdir as flowdir_m
from formosa.geomorphology._native import drainage_ridges as ridges_f
from formosa.geomorphology.drainage.directions import D8Directions
from formosa.geomorphology.drainage.metrics import (
    compute_dist2source,
    compute_flow_strahler_order,
)
from formosa.geomorphology.raster_validation import (
    validate_format_flowdirs,
    validate_format_freeform_coordinates,
    validate_format_valids,
)
from formosa.utils import Backend, NpCoords, NpFlowDir, raise_fortran_error


def compute_dist2conf_max(
    dirs: NDArray[NpFlowDir],
    valids: Optional[NDArray[np.bool_]] = None,
    x: Optional[NDArray[np.number]] = None,
    y: Optional[NDArray[np.number]] = None,
    dir_scheme: D8Directions = D8Directions(),
) -> NDArray[np.float32]:
    """
    Computes the maximum distance to confluence for each cell with
    its neighbours in the flow direction grid.

    If the cell does not share a confluence with any of its
    neighbours, the distance to sink is returned instead.
    This field can be used as a proxy for the ridge network, where
    cells with a larger distance to confluence are more likely to be
    part of the ridge network.

    Parameters
    ----------
    dirs : NDArray[uint8]
        Flow direction raster.
        - Expected shape: `(nrows, ncols)`.
    valids : NDArray[bool], optional
        Boolean mask indicating valid cells.
        If `None`, all cells are considered valid.
        - Expected shape: `(nrows, ncols)`, same as `dirs`.
        - Default mask is `None`.
    x : NDArray[float], optional
        X-coordinates of each cell.
        If `None`, cell column indices are used.
        - Expected shape: `(nrows, ncols)`, same as `dirs`.
        - Default input is `None`.
    y : NDArray[float], optional
        Y-coordinates of each cell.
        If `None`, cell row indices are used.
        - Expected shape: `(nrows, ncols)`, same as `dirs`.
        - Default input is `None`.
    dir_scheme : D8Directions, optional
        Instance of `D8Directions` defining the flow direction
        scheme.
        - Default scheme is `D8Directions()`.

    Returns
    -------
    bmax : NDArray[float32]
        Maximum distance to confluence for each cell.
        - Shape: `(nrows, ncols)`, same as `dirs`.

    Notes
    -----
    See :func:`compute_dist2ridge` for computing the distance to
    ridge based on this field.

    The Fortran backend represents the single-flow-direction raster
    as a forest: each valid cell has at most one downstream parent
    and each root is a sink. It computes parent, depth, sink, and
    cumulative-distance metadata once, then answers neighbouring-
    cell confluence queries using lowest-common-ancestor searches.
    A cyclic valid flow field is rejected rather than traversed
    indefinitely.

    Inputs are converted with :func:`numpy.asfortranarray` because
    the compiled routine consumes column-major arrays. Unlike an
    unconditional `astype`, this returns the original array when its
    dtype and layout already match, avoiding unnecessary full-grid
    copies.
    """
    dirs = validate_format_flowdirs(dirs)
    valids = validate_format_valids(valids, dirs, "flow direction raster")
    x, y = validate_format_freeform_coordinates(x, y, dirs.shape, np.float32)

    # Preserve already-compatible arrays. DEMGrid flow-direction and validity
    # arrays are commonly Fortran-contiguous, so unconditional astype calls here
    # would copy them despite requiring no representation change.
    bmax, err_code = ridges_f.compute_max_branch_dist(
        np.asfortranarray(dirs, dtype=np.uint8),
        np.asfortranarray(valids, dtype=bool),
        np.asfortranarray(x, dtype=np.float32),
        np.asfortranarray(y, dtype=np.float32),
        np.asfortranarray(dir_scheme.offsets, dtype=np.int32),
        np.asfortranarray(dir_scheme.codes, dtype=np.uint8),
    )
    raise_fortran_error("compute_max_branch_dist", err_code)
    # f2py normally returns the requested representation already; this is then
    # a no-copy normalization while still protecting the public dtype/layout.
    return np.asfortranarray(bmax, dtype=np.float32)


def compute_ridgedir(
    dirs: NDArray[NpFlowDir],
    dir_scheme: D8Directions = D8Directions(),
    valids: Optional[NDArray[np.bool_]] = None,
    x: Optional[NDArray[np.number]] = None,
    y: Optional[NDArray[np.number]] = None,
) -> NDArray[NpFlowDir]:
    """
    Computes flow directions over the inverted maximum-confluence field.

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
        Boolean mask indicating valid cells.
        If `None`, all cells are considered valid.
        - Expected shape: `(nrows, ncols)`, same as `dirs`.
        - Default mask is `None`.
    x : NDArray[float], optional
        X-coordinates of each cell.
        If `None`, cell column indices are used.
        - Expected shape: `(nrows, ncols)`, same as `dirs`.
        - Default input is `None`.
    y : NDArray[float], optional
        Y-coordinates of each cell.
        If `None`, cell row indices are used.
        - Expected shape: `(nrows, ncols)`, same as `dirs`.
        - Default input is `None`.

    Returns
    -------
    ridgedirs : NDArray[uint8]
        Flow directions along ridge paths.
        - Shape: `(nrows, ncols)`, same as `dirs`.
    """
    bmax = compute_dist2conf_max(dirs, valids=valids, x=x, y=y, dir_scheme=dir_scheme)
    bmaxdirs, _, _ = flowdir_m.compute_flowdir(
        -bmax, dir_scheme=dir_scheme, valids=valids, fill_depression=True
    )
    return bmaxdirs.astype(NpFlowDir, order="F")


def compute_dist2ridge(
    dirs: NDArray[NpFlowDir],
    dir_scheme: D8Directions = D8Directions(),
    valids: Optional[NDArray[np.bool_]] = None,
    x: Optional[NDArray[np.number]] = None,
    y: Optional[NDArray[np.number]] = None,
    dir_is_ridge: bool = False,
) -> NDArray[np.float32]:
    """
    Computes the 'distance to ridge' for each cell in the flow direction grid.

    The ridge network/intensity is defined as the maximum distance to confluence (see `compute_flow_dist2conf_max`), and the distance to ridge is computed as the downstream distance traversing the inverse of the intensity.

    Parameters
    ----------
    dirs : NDArray[uint8]
        Flow direction raster.
        - Expected shape: `(nrows, ncols)`.
    dir_scheme : D8Directions, optional
        Instance of `D8Directions` defining the flow direction scheme.
        Default scheme is `D8Directions()`.
    valids : NDArray[bool], optional
        Boolean mask indicating valid cells.
        If `None`, all cells are considered valid.
        - Expected shape: `(nrows, ncols)`, same as `dirs`.
        - Default mask is `None`.
    x : NDArray[float], optional
        X-coordinates of each cell.
        If `None`, cell column indices are used.
        - Expected shape: `(nrows, ncols)`, same as `dirs`.
        - Default input is `None`.
    y : NDArray[float], optional
        Y-coordinates of each cell.
        If `None`, cell row indices are used.
        - Expected shape: `(nrows, ncols)`, same as `dirs`.
        - Default input is `None`.
    dir_is_ridge : bool, optional
        Whether `dirs` is already a ridge flow direction grid.
        - Default option is `False`.

    Returns
    -------
    bmaxdists : NDArray[float32]
        Distance to ridge for each cell.
        - Shape: `(nrows, ncols)`, same as `dirs`.
    """
    dirs = validate_format_flowdirs(dirs)
    valids = validate_format_valids(valids, dirs, "flow direction raster")
    x, y = validate_format_freeform_coordinates(x, y, dirs.shape, np.float32)

    if dir_is_ridge:
        bmaxdirs = dirs
    else:
        bmaxdirs = compute_ridgedir(
            dirs, dir_scheme=dir_scheme, valids=valids, x=x, y=y
        )
    bmaxdists = compute_dist2source(
        bmaxdirs, dir_scheme=dir_scheme, x=x, y=y, valids=valids
    )
    return bmaxdists


def compute_ridge_strahler_order(
    dirs: NDArray[NpFlowDir],
    dir_scheme: D8Directions = D8Directions(),
    valids: Optional[NDArray[np.bool_]] = None,
    indegs: Optional[NDArray[np.integer]] = None,
    backend: Backend = "fortran",
    dir_is_ridge: bool = False,
) -> NDArray[np.uint8]:
    """
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
    indegs : NDArray[int], optional
        In-degree (number of upstream cells) for each cell.
        If `None`, it will be computed from the ridge flow
        directions.
        - Expected shape: `(nrows, ncols)`, same as `dirs`.
        - Default input is `None`.
    backend : {'fortran', 'python'}, optional
        Backend to use for computation.
        `'fortran'` uses the FORTRAN extension for performance,
        while `'python'` uses a pure Python implementation.
        Default backend is `'fortran'`.

    Returns
    -------
    orders : NDArray[uint8]
        Ridge Strahler order for each cell.
        - Shape: `(nrows, ncols)`, same as `dirs`.
    """
    dirs = validate_format_flowdirs(dirs)

    if dir_is_ridge:
        bmaxdirs = dirs
    else:
        bmaxdirs = compute_ridgedir(dirs, dir_scheme=dir_scheme, valids=valids)
    orders = compute_flow_strahler_order(
        bmaxdirs, dir_scheme=dir_scheme, valids=valids, indegs=indegs, backend=backend
    )
    return orders.astype(np.uint8, order="F")
