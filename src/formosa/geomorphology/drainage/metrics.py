"""
Computes cell-level geomorphological metrics from raster flow
directions.

Last modified: 2026-08-23, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np

from formosa.geomorphology.drainage.directions import D8Directions
from formosa.geomorphology.drainage.flowdir import count_indegree
from formosa.geomorphology._native import drainage_metrics as metrics_f
import formosa.geomorphology.drainage._backends.metrics_py as metrics_py
from formosa.utils import NpFlowDir
from formosa.utils import Backend, raise_fortran_error
from formosa.geomorphology._validation import (
    validate_2d_array,
    validate_same_shape,
    validate_format_valids,
    validate_format_flowdirs,
)


from typing import Optional
from numpy.typing import NDArray


def compute_flow_accumulation(
    dirs: NDArray[NpFlowDir],
    valids: Optional[NDArray[np.bool_]] = None,
    weights: Optional[NDArray[np.floating]] = None,
    indegs: Optional[NDArray[np.integer]] = None,
    dsij: Optional[NDArray[np.integer]] = None,
    dir_scheme: D8Directions = D8Directions(),
    backend: Backend = "fortran",
) -> NDArray[np.float32]:
    """
    Computes flow accumulation for each cell in a flow direction grid.

    Parameters
    ----------
    dirs : NDArray[uint8]
        A 2D array representing the flow directions for each cell.
    valids : NDArray[bool], optional
        A boolean mask array indicating valid cells in the flow direction grid.
        If `None`, all cells are considered valid.
        Default is `None`.
    weights : NDArray[float], optional
        A 2D array of weights for each cell, representing the contribution of each cell to its downstream cell.
        If `None`, each valid cell contributes a weight of 1.0.
        Default is `None`.
    indegs : NDArray[int], optional
        A 2D array representing the indegree (number of upstream cells) for each cell.
        If `None`, `indegs` are computed from the flow direction grid.
        Default is `None`.
    dsij : NDArray[int], optional
        A 2D array of downstream cell indices for each cell.
        If `None`, downstream indices are computed from the flow direction grid.
        Default is `None`.
    dir_scheme : D8Directions, optional
        An instance of `D8Directions` defining the flow direction scheme.
        Default is `D8Directions()`.
    backend : {'fortran', 'python'}, optional
        Backend to use for computation.
        `'fortran'` uses the Fortran extension for performance,
        while `'python'` uses a pure Python implementation.
        Default backend is `'fortran'`.

    Returns
    -------
    accums : NDArray[float32]
        A 2D array representing the flow accumulation for each cell.
    """
    match backend:
        case "python":
            accums = metrics_py.compute_flow_accumulation(
                dirs,
                valids=valids,
                weights=weights,
                indegs=indegs,
                dsij=dsij,
                dir_scheme=dir_scheme,
            )
        case "fortran":
            if indegs is None:
                indegs = count_indegree(dirs, dir_scheme=dir_scheme)
            else:
                validate_2d_array(indegs, "in-degree raster")

            if valids is None:
                valids = np.ones(dirs.shape, dtype=bool)
            else:
                validate_format_valids(valids, indegs)

            if weights is None:
                weights = np.where(valids, 1.0, 0.0).astype(np.float32)
                validate_same_shape(weights, indegs, "weights", "in-degree raster")

            accums, err_code = metrics_f.compute_flow_accumulation(
                dirs.astype(np.uint8, order="F"),
                valids.astype(bool, order="F"),
                weights.astype(np.float32, order="F"),
                indegs.astype(np.int8, order="F"),
                dir_scheme.offsets.astype(np.int32, order="F"),
                dir_scheme.codes.astype(np.uint8, order="F"),
            )
            raise_fortran_error("compute_flow_accumulation", err_code)

    return accums.astype(np.float32, order="F")


def compute_flow_strahler_order(
    dirs: NDArray[NpFlowDir],
    dir_scheme: D8Directions = D8Directions(),
    valids: Optional[NDArray[np.bool_]] = None,
    indegs: Optional[NDArray[np.integer]] = None,
    backend: Backend = "fortran",
) -> NDArray[np.uint8]:
    """
    Computes the Strahler order for each cell in a flow direction grid.

    Parameters
    ----------
    dirs : NDArray[uint8]
        2D array representing the flow directions for each cell.
    dir_scheme : D8Directions, optional
        Instance of `D8Directions` defining the flow direction scheme.
        Default is `D8Directions()`.
    valids : NDArray[bool], optional
        Boolean mask array indicating valid cells in the flow direction grid.
        If `None`, all cells are considered valid.
        Default is `None`.
    indegs : NDArray[int], optional
        2D array representing the number of upstream cells for each cell.
        If `None`, it will be computed from the flow direction grid.
        Default is `None`.
    backend : {'fortran', 'python'}, optional
        Backend to use for computation.
        `'fortran'` uses the Fortran extension for performance,
        while `'python'` uses a pure Python implementation.
        Default backend is `'fortran'`.

    Returns
    -------
    orders : NDArray[uint8]
        2D integer array representing the Strahler order for each cell.
        Invalid cells will have a Strahler order of 0.

    Raises
    ------
    AssertionError
        If the input have the wrong types or shapes.
    """

    dirs = validate_format_flowdirs(dirs)
    valids = validate_format_valids(valids, dirs, "flow direction raster")

    if indegs is None:
        indegs = count_indegree(dirs, dir_scheme, valids=valids, backend=backend)
    else:
        validate_same_shape(indegs, dirs, "in-degree", "flow direction rasters")

    match backend:
        case "python":
            orders = metrics_py.compute_flow_strahler_order(
                dirs=dirs, dir_scheme=dir_scheme, valids=valids, indegs=indegs
            )
        case "fortran":
            orders, err_code = metrics_f.compute_flow_strahler_order(
                dirs.astype(np.uint8, order="F"),
                valids.astype(bool, order="F"),
                indegs.astype(np.int8, order="F"),
                dir_scheme.offsets.astype(np.int32, order="F"),
                dir_scheme.codes.astype(np.uint8, order="F"),
            )
            raise_fortran_error("compute_flow_strahler_order", err_code)
    orders[~valids] = 0
    return orders.astype(np.uint8, order="F")


def compute_dist2source(
    dirs: NDArray[NpFlowDir],
    dir_scheme: D8Directions = D8Directions(),
    x: Optional[NDArray[np.number]] = None,
    y: Optional[NDArray[np.number]] = None,
    valids: Optional[NDArray[np.bool_]] = None,
    indegs: Optional[NDArray[np.integer]] = None,
) -> NDArray[np.float32]:
    """
    Computes the distance downstream along flow directions for each cell in the flow direction grid.

    Parameters
    ----------
    dirs : NDArray[uint8]
        A 2D array representing the flow direction for each cell.
    dir_scheme : D8Directions, optional
        An instance of `D8Directions` defining the flow direction scheme.
        Default is `D8Directions()`.
    x : NDArray[int | float], optional
        A 2D array representing the x-coordinates of each cell. If `None`, cell indices are used.
        Default is `None`.
    y : NDArray[int | float], optional
        A 2D array representing the y-coordinates of each cell. If `None`, cell indices are used.
        Default is `None`.
    valids : NDArray[bool], optional
        A boolean mask array indicating valid cells in the flow direction grid.
        If `None`, all cells are considered valid.
        Default is `None`.
    indegs : NDArray[int], optional
        A 2D array representing the indegree (number of upstream cells) for each cell.
        If `None`, indegs are computed from the flow direction grid.
        Default is `None`.

    Returns
    -------
    dists : NDArray[float32]
        A 2D array representing the downstream distance for each cell.

    Raises
    ------
    TypeError
        If the input arrays are not of the expected type or format.
    ValueError
        If the shapes of the input arrays do not match the expected dimensions.
    """
    validate_format_flowdirs(dirs)
    valids = validate_format_valids(valids, dirs, "flow direction raster")
    if x is not None and y is not None:
        validate_same_shape(x, dirs, "X coordinates", "flow direction raster")
        validate_same_shape(y, dirs, "Y coordinates", "flow direction raster")
    else:
        x = np.arange(dirs.shape[1], dtype=np.float32)
        y = np.arange(dirs.shape[0], dtype=np.float32)
        x, y = np.meshgrid(x, y, indexing="xy")
    if indegs is None:
        indegs = count_indegree(dirs, dir_scheme=dir_scheme)
    else:
        validate_same_shape(indegs, dirs, "in-degree", "flow direction rasters")

    dists, err_code = metrics_f.compute_dist2source(
        dirs.astype(np.uint8, order="F"),
        valids.astype(bool, order="F"),
        x.astype(np.float32, order="F"),
        y.astype(np.float32, order="F"),
        indegs.astype(np.int8, order="F"),
        dir_scheme.offsets.astype(np.int32, order="F"),
        dir_scheme.codes.astype(np.uint8, order="F"),
    )
    raise_fortran_error("compute_dist2source", err_code)
    return dists.astype(np.float32, order="F")


def compute_dist2sink(
    dirs: NDArray[NpFlowDir],
    dir_scheme: D8Directions = D8Directions(),
    x: Optional[NDArray[np.number]] = None,
    y: Optional[NDArray[np.number]] = None,
    valids: Optional[NDArray[np.bool_]] = None,
) -> NDArray[np.float32]:
    """
    Computes the distance upstream along flow directions for each cell in the flow direction grid.

    Parameters
    ----------
    dirs : NDArray[uint8]
        A 2D array representing the flow direction for each cell.
    dir_scheme : D8Directions, optional
        An instance of `D8Directions` defining the flow direction scheme.
        Default is `D8Directions()`.
    x : NDArray[int | float], optional
        A 2D array representing the x-coordinates of each cell. If `None`, a default grid will be created.
    y : NDArray[int | float], optional
        A 2D array representing the y-coordinates of each cell. If `None`, a default grid will be created.
    valids : NDArray[bool], optional
        A boolean mask array where `True` indicates valid cells. If `None`, all non-NaN cells in `dirs` are considered valid.

    Returns
    -------
    dists : NDArray[float32]
        A 2D array representing the upstream distance for each cell.
    """
    validate_format_flowdirs(dirs)
    valids = validate_format_valids(valids, dirs, "flow direction raster")
    if x is not None and y is not None:
        validate_same_shape(x, dirs, "X coordinates", "flow direction raster")
        validate_same_shape(y, dirs, "Y coordinates", "flow direction raster")
    else:
        x = np.arange(dirs.shape[1], dtype=np.float32)
        y = np.arange(dirs.shape[0], dtype=np.float32)
        x, y = np.meshgrid(x, y, indexing="xy")

    dists, err_code = metrics_f.compute_dist2sink(
        dirs.astype(np.uint8, order="F"),
        x.astype(np.float32, order="F"),
        y.astype(np.float32, order="F"),
        valids.astype(bool, order="F"),
        dir_scheme.offsets.astype(np.int32, order="F"),
        dir_scheme.codes.astype(np.uint8, order="F"),
    )
    raise_fortran_error("compute_dist2sink", err_code)
    return dists.astype(np.float32, order="F")
