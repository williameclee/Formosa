"""
Computes cell-level geomorphological metrics from raster flow
directions.

Created: 2026-08-01, En-Chi Lee (williameclee@gmail.com)
Last modified: 2026-08-23, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np
from numpy.typing import NDArray

from formosa.geomorphology._native import drainage_metrics as metrics_f
from formosa.geomorphology.drainage._backends import metrics_py
from formosa.geomorphology.drainage.directions import D8Directions
from formosa.geomorphology.drainage.flowdir import count_indegree
from formosa.geomorphology.drainage.neighbours import compute_downstream_indices
from formosa.geomorphology.raster_validation import (
    validate_format_dir_scheme,
    validate_format_flowdirs,
    validate_format_freeform_coordinates,
    validate_format_valids,
)
from formosa.utils import Backend, NpCoords, NpFlowDir, raise_fortran_error
from formosa.utils.validation import validate_same_shape


def compute_flow_accumulation(
    dirs: NDArray[NpFlowDir],
    dir_scheme: D8Directions | None = None,
    valids: NDArray[np.bool_] | None = None,
    weights: NDArray[np.floating] | None = None,
    indegs: NDArray[np.integer] | None = None,
    dsij: NDArray[np.integer] | None = None,
    backend: Backend = "fortran",
) -> NDArray[np.float64]:
    """
    Computes flow accumulation for each cell in a flow direction
    grid.

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
    weights : NDArray[float], optional
        Weights for each cell, representing the contribution of each
        cell to its downstream cell.
        If `None`, each valid cell contributes a weight of 1.0.
        - Expected shape: `(nrows, ncols)`, same as `dirs`.
        - Default input is `None`.
    indegs : NDArray[int], optional
        In-degree (number of upstream cells) for each cell.
        If `None`, `indegs` are computed from the flow direction
        grid.
        - Expected shape: `(nrows, ncols)`, same as `dirs`.
        - Default input is `None`.
    dsij : NDArray[int], optional
        Flattened downstream cell indices for each cell.
        If `None`, downstream indices are computed from the flow
        direction grid.
        - Expected shape: `(nrows, ncols)`, same as `dirs`.
        - Default input is `None`.
    backend : {'fortran', 'python'}, optional
        Backend to use for computation.
        `'fortran'` uses the Fortran extension for performance,
        while `'python'` uses a pure Python implementation.
        - Default backend is `'fortran'`.

    Returns
    -------
    accums : NDArray[float64]
        Accumulated flow weights for each cell.
        - Shape: `(nrows, ncols)`, same as `dirs`.
    """
    dirs = validate_format_flowdirs(dirs)
    dir_scheme = validate_format_dir_scheme(dir_scheme)
    valids = validate_format_valids(valids, dirs, "flow direction raster")

    if weights is not None:
        validate_same_shape(weights, dirs, "weights", "flow direction rasters")
    else:
        weights = np.ones_like(dirs, dtype=np.float64)

    if indegs is None:
        indegs = count_indegree(dirs, dir_scheme, valids=valids, backend=backend)
    else:
        validate_same_shape(indegs, dirs, "in-degree", "flow direction rasters")

    if dsij is None:
        _, _, dsij, _ = compute_downstream_indices(
            dirs, dir_scheme=dir_scheme, valids=valids, check=False
        )
        assert dsij is not None
    else:
        validate_same_shape(
            dsij, dirs, "downstream cell indices", "flow direction rasters"
        )
    match backend:
        case "python":
            accums = metrics_py.compute_flow_accumulation(
                dirs=dirs,
                weights=weights,
                indegs=indegs,
                dsij=dsij,
                dir_scheme=dir_scheme,
                valids=valids,
            )
        case "fortran":
            accums, err_code = metrics_f.compute_flow_accumulation(
                dirs.astype(np.uint8, order="F"),
                valids.astype(bool, order="F"),
                weights.astype(np.float64, order="F"),
                indegs.astype(np.int8, order="F"),
                dsij.astype(np.int32, order="F"),
                dir_scheme.offsets.astype(np.int32, order="F"),
                dir_scheme.codes.astype(np.uint8, order="F"),
            )
            raise_fortran_error("compute_flow_accumulation", err_code)
    accums[~valids] = 0
    return accums.astype(np.float64, order="F")


def compute_flow_strahler_order(
    dirs: NDArray[NpFlowDir],
    dir_scheme: D8Directions | None = None,
    valids: NDArray[np.bool_] | None = None,
    indegs: NDArray[np.integer] | None = None,
    backend: Backend = "fortran",
) -> NDArray[np.uint8]:
    """
    Computes Strahler stream order for each cell based on flow
    direction.

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
        If `None`, it will be computed from the flow direction grid.
        - Expected shape: `(nrows, ncols)`, same as `dirs`.
        - Default input is `None`.
    backend : {'fortran', 'python'}, optional
        Backend to use for computation.
        `'fortran'` uses the Fortran extension for performance,
        while `'python'` uses a pure Python implementation.
        - Default backend is `'fortran'`.

    Returns
    -------
    orders : NDArray[uint8]
        Strahler order for each cell.
        Invalid cells will have a Strahler order of 0.
        - Shape: `(nrows, ncols)`, same as `dirs`.
    """

    dirs = validate_format_flowdirs(dirs)
    dir_scheme = validate_format_dir_scheme(dir_scheme)
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
    dir_scheme: D8Directions | None = None,
    x: NDArray[NpCoords] | None = None,
    y: NDArray[NpCoords] | None = None,
    valids: NDArray[np.bool_] | None = None,
    indegs: NDArray[np.integer] | None = None,
) -> NDArray[np.float32]:
    """
    Computes the distance downstream along flow directions for each cell in the flow direction grid.

    Parameters
    ----------
    dirs : NDArray[uint8]
        Flow direction raster.
        - Expected shape: `(nrows, ncols)`.
    dir_scheme : D8Directions, optional
        Instance of `D8Directions` defining the flow direction
        scheme.
        - Default scheme is `D8Directions()`.
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
    valids : NDArray[bool], optional
        Boolean mask indicating valid cells in the flow direction
        grid.
        If `None`, all cells are considered valid.
        - Expected shape: `(nrows, ncols)`, same as `dirs`.
        - Default mask is `None`.
    indegs : NDArray[int], optional
        In-degree (number of upstream cells) for each cell.
        If `None`, indegs are computed from the flow direction grid.
        - Expected shape: `(nrows, ncols)`, same as `dirs`.
        - Default input is `None`.

    Returns
    -------
    dists : NDArray[float32]

    Raises
    ------
    TypeError
        If the input arrays are not of the expected type or format.
    ValueError
        If the shapes of the input arrays do not match the expected dimensions.
        Downstream distance for each cell.
        - Shape: `(nrows, ncols)`, same as `dirs`.
    """
    dirs = validate_format_flowdirs(dirs)
    dir_scheme = validate_format_dir_scheme(dir_scheme)
    valids = validate_format_valids(valids, dirs, "flow direction raster")
    x, y = validate_format_freeform_coordinates(x, y, dirs.shape, np.float32)
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
    dir_scheme: D8Directions | None = None,
    x: NDArray[NpCoords] | None = None,
    y: NDArray[NpCoords] | None = None,
    valids: NDArray[np.bool_] | None = None,
) -> NDArray[np.float32]:
    """
    Computes the distance upstream along flow directions for each cell in the flow direction grid.

    Parameters
    ----------
    dirs : NDArray[uint8]
        Flow direction raster.
        - Expected shape: `(nrows, ncols)`.
    dir_scheme : D8Directions, optional
        Instance of `D8Directions` defining the flow direction
        scheme.
        - Default scheme is `D8Directions()`.
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
    valids : NDArray[bool], optional
        Boolean mask indicating valid cells in the flow direction
        grid.
        If `None`, all cells are considered valid.
        - Expected shape: `(nrows, ncols)`, same as `dirs`.
        - Default mask is `None`.

    Returns
    -------
    dists : NDArray[float32]
        Upstream distance for each cell.
        - Shape: `(nrows, ncols)`, same as `dirs`.
    """
    dirs = validate_format_flowdirs(dirs)
    dir_scheme = validate_format_dir_scheme(dir_scheme)
    valids = validate_format_valids(valids, dirs, "flow direction raster")
    x, y = validate_format_freeform_coordinates(x, y, dirs.shape, np.float32)

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
