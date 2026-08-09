# Last modified
#   2026-02-11, En-Chi Lee (williameclee@arizona.edu)
#     - Rename flowdir functions to be more descriptive.
#   2026-06-09, En-Chi Lee (williameclee@gmail.com)
#     - Added `compute_flow_dist2ridge` function to compute
#       'distance to ridges'.
#     - Added error for missing FORTRAN backend.
#     - Removed NumPy type `np.bool` to either `np.bool_` or `bool`
#       for compatibility with newer NumPy versions.
#   2026-06-10, En-Chi Lee (williameclee@gmail.com)
#     - Small refactors and documentation cleanup.
#   2026-06-11, En-Chi Lee (williameclee@gmail.com)
#     - Moved Python backend implementations and auxiliary functions
#       to separate files.
#     - Standardised variable, argument, and function names.
#   2026-06-30, En-Chi Lee (williameclee@gmail.com)
#     - Added `x` and `y` into `compute_dist2source` in
#       `compute_dist2ridge`.
#     - Added functions `compute_ridgedir` and
#       `compute_ridge_strahler_order`.
#   2026-07-14, En-Chi Lee (williameclee@gmail.com)
#     - Splitted `geomorphology.flowdir` into submodules.
#   2026-08-04, En-Chi Lee (williameclee@gmail.com)
#     - Updated `compute_dist2conf_max` and related functions'
#       interface to reflect FORTRAN backend changes.

import numpy as np

from formosa.utils import Backend, raise_fortran_error
from formosa.geomorphology.drainage.directions import D8Directions
import formosa.geomorphology.drainage.flowdir as flowdir_m
from formosa.geomorphology.drainage.metrics import (
    compute_dist2source,
    compute_flow_strahler_order,
)
from formosa.geomorphology.drainage_f import drainage_ridges as ridges_f

from typing import Optional
import numpy.typing as npt


def compute_dist2conf_max(
    dirs: npt.NDArray[np.integer],
    valids: Optional[npt.NDArray[np.bool_]] = None,
    x: Optional[npt.NDArray[np.number]] = None,
    y: Optional[npt.NDArray[np.number]] = None,
    dir_scheme: D8Directions = D8Directions(),
) -> npt.NDArray[np.float32]:
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
        2D array representing the flow directions for each cell.
    valids : NDArray[bool], optional
        Boolean mask array where `True` indicates valid cells. If `None`, all cells are considered valid.
        Default input is `None`.
    x : NDArray[int | float], optional
        2D array representing the x-coordinates of each cell. If `None`, a default grid will be created.
        Default input is `None`.
    y : NDArray[int | float], optional
        2D array representing the y-coordinates of each cell. If `None`, a default grid will be created.
        Default input is `None`.
    dir_scheme : D8Directions, optional
        Instance of `D8Directions` defining the flow direction scheme.
        Default scheme is `D8Directions()`.

    Returns
    -------
    NDArray[float32]
        2D array representing the maximum distance to confluence for
        each cell.

    Notes
    -----
    See :func:`compute_dist2ridge` for computing the distance to
    ridge based on this field.

    The FORTRAN backend represents the single-flow-direction raster
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
    if valids is None:
        valids = np.ones(dirs.shape, dtype=bool)
    elif isinstance(valids, np.ndarray):
        assert (
            valids.shape == dirs.shape
        ), f"Shape for flow direction ({dirs.shape}) and valid mask ({valids.shape}) do not match."
    else:
        raise TypeError(f"Valid mask must be a NumPy array (got {type(valids)}).")
    if x is not None and y is not None:
        assert (
            x.shape == dirs.shape and y.shape == dirs.shape
        ), f"Shapes for flow direction ({dirs.shape}) and x ({x.shape}) and y ({y.shape}) must match."
    else:
        x = np.arange(dirs.shape[1], dtype=np.float32)
        y = np.arange(dirs.shape[0], dtype=np.float32)
        x, y = np.meshgrid(x, y, indexing="xy")

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
    dirs: npt.NDArray[np.integer],
    dir_scheme: D8Directions = D8Directions(),
    valids: Optional[npt.NDArray[np.bool_]] = None,
    x: Optional[npt.NDArray[np.number]] = None,
    y: Optional[npt.NDArray[np.number]] = None,
) -> npt.NDArray[np.uint8]:
    bmax = compute_dist2conf_max(
        dirs,
        valids=valids,
        x=x,
        y=y,
        dir_scheme=dir_scheme,
    )
    bmaxdirs, _, _ = flowdir_m.compute_flowdir(
        -bmax, dir_scheme=dir_scheme, valids=valids, fill_depression=True
    )
    return bmaxdirs.astype(np.uint8, order="F")


def compute_dist2ridge(
    dirs: npt.NDArray[np.integer],
    dir_scheme: D8Directions = D8Directions(),
    valids: Optional[npt.NDArray[np.bool_]] = None,
    x: Optional[npt.NDArray[np.number]] = None,
    y: Optional[npt.NDArray[np.number]] = None,
    dir_is_ridge: bool = False,
) -> npt.NDArray[np.float32]:
    """
    Computes the 'distance to ridge' for each cell in the flow direction grid.

    The ridge network/intensity is defined as the maximum distance to confluence (see `compute_flow_dist2conf_max`), and the distance to ridge is computed as the downstream distance traversing the inverse of the intensity.

    Parameters
    ----------
    dirs : NDArray[int]
        2D array representing the flow directions for each cell.
    dir_scheme : D8Directions, optional
        Instance of `D8Directions` defining the flow direction scheme.
        Default scheme is `D8Directions()`.
    valids : NDArray[bool], optional
        Boolean mask array where `True` indicates valid cells.
        If `None`, all cells are considered valid.
        Default input is `None`.
    x : NDArray[int | float], optional
        2D array representing the x-coordinates of each cell.
        If `None`, a default grid will be created.
        Default input is `None`.
    y : NDArray[int | float], optional
        2D array representing the y-coordinates of each cell.
        If `None`, a default grid will be created.
        Default input is `None`.

    Returns
    -------
    bmaxdists : NDArray[float32]
        2D array representing the distance to ridge for each cell.
    """
    if dir_is_ridge:
        bmaxdirs = dirs
    else:
        bmaxdirs = compute_ridgedir(
            dirs,
            dir_scheme=dir_scheme,
            valids=valids,
            x=x,
            y=y,
        )
    bmaxdists = compute_dist2source(
        bmaxdirs, dir_scheme=dir_scheme, x=x, y=y, valids=valids
    )
    return bmaxdists


def compute_ridge_strahler_order(
    dirs: npt.NDArray[np.integer],
    dir_scheme: D8Directions = D8Directions(),
    valids: Optional[npt.NDArray[np.bool_]] = None,
    indegs: Optional[npt.NDArray[np.integer]] = None,
    backend: Backend = "fortran",
    dir_is_ridge: bool = False,
) -> npt.NDArray[np.uint8]:
    """
    Parameters
    ----------
    backend : {'fortran', 'python'}, optional
        Backend to use for computation.
        `'fortran'` uses the FORTRAN extension for performance,
        while `'python'` uses a pure Python implementation.
        Default backend is `'fortran'`.
    """

    if dir_is_ridge:
        bmaxdirs = dirs
    else:
        bmaxdirs = compute_ridgedir(
            dirs,
            dir_scheme=dir_scheme,
            valids=valids,
        )
    orders = compute_flow_strahler_order(
        bmaxdirs,
        dir_scheme=dir_scheme,
        valids=valids,
        indegs=indegs,
        backend=backend,
    )
    return orders.astype(np.uint8, order="F")
