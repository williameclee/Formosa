# Last modified
#   2026-02-11, En-Chi Lee (williameclee@arizona.edu)
#     - Rename flowdir functions to be more descriptive.
#   2026-06-09, En-Chi Lee (williameclee@gmail.com)
#     - Added error for missing FORTRAN backend.
#     - Removed NumPy type `np.bool` to either `np.bool_` or `bool`
#       for compatibility with newer NumPy versions.
#     - Renamed FORTRAN function call: `compute_masked_flowdir` ->
#       `compute_synthetic_flowdir`.
#     - Added `valids` argument to `label_flats` function.
#   2026-06-10, En-Chi Lee (williameclee@gmail.com)
#     - Small refactors and documentation cleanup.
#   2026-06-11, En-Chi Lee (williameclee@gmail.com)
#     - Moved Python backend implementations and auxiliary functions
#       to separate files.
#     - Standardised variable, argument, and function names.
#   2026-06-30, En-Chi Lee (williameclee@gmail.com)
#     - Changed strahler order output to 8-bit unsigned integer.
#   2026-07-01, En-Chi Lee (williameclee@gmail.com)
#     - Allowed specifying validity mask in `count_indegree`.
#     - Added function `construct_flowgraph`.
#   2026-07-08, En-Chi Lee (williameclee@gmail.com)
#     - Renamed helper submodule from `aux` to `utils`.
#   2026-07-14, En-Chi Lee (williameclee@gmail.com)
#     - Splitted `geomorphology.flowdir` into submodules.
#   2026-07-30, En-Chi Lee (williameclee@gmail.com)
#     - Fixed Python/FORTRAN backend behaviour parity in
#       `compute_flow_strahler_order`.
#   2026-08-03, En-Chi Lee (williameclee@gmail.com)
#     - Implemented functions `find_acyclic_flowdirs` and
#       `find_cyclic_flowdirs` with both FORTRAN and Python
#       backends.


import numpy as np

from formosa.geomorphology.drainage.directions import D8Directions
from formosa.geomorphology.drainage.utils import raise_fortran_error
from formosa.geomorphology.drainage.flowdir import count_indegree
import formosa.geomorphology.drainage._backends.watersheds_py as wshed_py
from formosa.geomorphology.drainage_f import flowdir_watersheds as wshed_f

from typing import Literal, Optional
import numpy.typing as npt


def compute_flow_accumulation(
    dirs: npt.NDArray[np.integer],
    valids: Optional[npt.NDArray[np.bool_]] = None,
    weights: Optional[npt.NDArray[np.floating]] = None,
    indegs: Optional[npt.NDArray[np.integer]] = None,
    dsij: Optional[npt.NDArray[np.integer]] = None,
    dir_scheme: D8Directions = D8Directions(),
    backend: Literal["fortran", "python"] = "fortran",
) -> npt.NDArray[np.float32]:
    """
    Computes flow accumulation for each cell in a flow direction grid.

    Parameters
    ----------
    dirs : NDArray[int]
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
        The backend to use for computation. 'fortran' uses the Fortran extension for performance, while 'python' uses a pure Python implementation.
        Default is 'fortran'.

    Returns
    -------
    accums : NDArray[float32]
        A 2D array representing the flow accumulation for each cell.
    """
    match backend:
        case "python":
            accums = wshed_py.compute_flow_accumulation(
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

            if valids is None:
                valids = np.ones(dirs.shape, dtype=bool)

            if weights is None:
                weights = np.where(valids, 1.0, 0.0).astype(np.float32)

            accums, err_code = wshed_f.compute_flow_accumulation(
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
    dirs: npt.NDArray[np.integer],
    dir_scheme: D8Directions = D8Directions(),
    valids: Optional[npt.NDArray[np.bool_]] = None,
    indegs: Optional[npt.NDArray[np.integer]] = None,
    backend: Literal["fortran", "python"] = "fortran",
) -> npt.NDArray[np.uint8]:
    """
    Computes the Strahler order for each cell in a flow direction grid.

    Parameters
    ----------
    dirs : NDArray[int], optional
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
        Backend to use for computation
        `'fortran'` uses the FORTRAN extension for performance, while 'python' uses a pure Python implementation.
        Default option is `'fortran'`.

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

    assert (
        dirs.ndim == 2
    ), f"Flow directions must be a 2D array, got shape {dirs.shape}."

    if valids is None:
        valids = np.ones(dirs.shape, dtype=bool, order="F")
    else:
        assert isinstance(
            valids, np.ndarray
        ), f"Valid mask must be a NumPy array (got {type(valids)})."
        assert (
            valids.shape == dirs.shape
        ), f"Shape for flow direction ({dirs.shape}) and valid mask ({valids.shape}) do not match."
        valids = valids.astype(bool, order="F", copy=False)

    if indegs is None:
        indegs = count_indegree(
            dirs, dir_scheme=dir_scheme, valids=valids, backend=backend
        )
    else:
        assert isinstance(
            indegs, np.ndarray
        ), f"Indegree must be a NumPy array (got {type(indegs)})."
        assert (
            indegs.shape == dirs.shape
        ), f"Shape for flow direction ({dirs.shape}) and indegree ({indegs.shape}) do not match."

    match backend:
        case "python":
            orders = wshed_py.compute_flow_strahler_order(
                dirs=dirs, dir_scheme=dir_scheme, valids=valids, indegs=indegs
            )
        case "fortran":
            orders, err_code = wshed_f.compute_flow_strahler_order(
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
    dirs: npt.NDArray[np.integer],
    dir_scheme: D8Directions = D8Directions(),
    x: Optional[npt.NDArray[np.number]] = None,
    y: Optional[npt.NDArray[np.number]] = None,
    valids: Optional[npt.NDArray[np.bool_]] = None,
    indegs: Optional[npt.NDArray[np.integer]] = None,
) -> npt.NDArray[np.float32]:
    """
    Computes the distance downstream along flow directions for each cell in the flow direction grid.

    Parameters
    ----------
    dirs : NDArray[int]
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
    if indegs is None:
        indegs = count_indegree(dirs, dir_scheme=dir_scheme)
    elif isinstance(indegs, np.ndarray):
        assert (
            indegs.shape == dirs.shape
        ), f"Shape for flow direction ({dirs.shape}) and indegree ({indegs.shape}) do not match."
    else:
        raise TypeError(f"Indegree must be a NumPy array (got {type(indegs)}).")

    dists, err_code = wshed_f.compute_dist2source(
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


def label_watersheds(
    dirs: npt.NDArray[np.integer],
    dir_scheme: D8Directions = D8Directions(),
    valids: Optional[npt.NDArray[np.bool_]] = None,
    backend: Literal["fortran", "python"] = "fortran",
) -> npt.NDArray[np.int32]:
    """
    Finds and labels watersheds in a DEM based on flow direction.

    Parameters
    ----------
    dirs : NDArray[int]
        A 2D array representing the flow direction for each cell.
    dir_scheme : D8Directions, optional
        An instance of `D8Directions` defining the flow direction scheme.
        Default is `D8Directions()`.
    valids : NDArray[bool], optional
        A boolean mask array indicating valid cells in the flow direction grid.
        If `None`, all non-NaN cells in flowdirs are considered valid.
        Default is `None`.

    Returns
    -------
    watersheds : NDArray[int32]
        A 2D array where each watershed is labeled with a unique integer.
    """
    match backend:
        case "python":
            watersheds = wshed_py.label_watersheds(
                dirs=dirs,
                dir_scheme=dir_scheme,
                valids=valids,
            )
        case "fortran":
            if valids is None:
                valids = np.ones(dirs.shape, dtype=bool)
            elif isinstance(valids, np.ndarray):
                assert (
                    valids.shape == dirs.shape
                ), f"Shape for flow direction ({dirs.shape}) and valid mask ({valids.shape}) do not match."
                valids = valids.astype(bool, copy=False) & (~np.isnan(dirs))
                dirs = np.where(valids, dirs, np.nan)
            else:
                raise TypeError(
                    f"Valid mask must be a NumPy array (got {type(valids)})."
                )

            watersheds, err_code = wshed_f.label_watersheds(
                dirs.astype(np.uint8, order="F"),
                valids.astype(bool, order="F"),
                dir_scheme.offsets.astype(np.int32, order="F"),
                dir_scheme.codes.astype(np.uint8, order="F"),
            )
            raise_fortran_error("label_watersheds", err_code)
    return watersheds.astype(np.int32, order="F")


def compute_dist2sink(
    dirs: npt.NDArray[np.integer],
    dir_scheme: D8Directions = D8Directions(),
    x: Optional[npt.NDArray[np.number]] = None,
    y: Optional[npt.NDArray[np.number]] = None,
    valids: Optional[npt.NDArray[np.bool_]] = None,
) -> npt.NDArray[np.float32]:
    """
    Computes the distance upstream along flow directions for each cell in the flow direction grid.

    Parameters
    ----------
    dirs : NDArray[int]
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
    if valids is None:
        valids = ~np.isnan(dirs)
    elif isinstance(valids, np.ndarray):
        assert (
            valids.shape == dirs.shape
        ), f"Shape for flow direction ({valids.shape}) and valid mask ({dirs.shape}) do not match."
        valids = valids.astype(bool, copy=False) & (~np.isnan(dirs))
        dirs = np.where(valids, dirs, np.nan)
    else:
        raise TypeError(
            f"Validity mask must be either None or a numpy array, (got {type(valids)})."
        )
    if x is not None and y is not None:
        assert (
            x.shape == dirs.shape and y.shape == dirs.shape
        ), f"Shapes for flow direction ({dirs.shape}) and x ({x.shape}) and y ({y.shape}) must match."
    else:
        x = np.arange(dirs.shape[1], dtype=np.float32)
        y = np.arange(dirs.shape[0], dtype=np.float32)
        x, y = np.meshgrid(x, y, indexing="xy")

    dists, err_code = wshed_f.compute_dist2sink(
        dirs.astype(np.uint8, order="F"),
        x.astype(np.float32, order="F"),
        y.astype(np.float32, order="F"),
        valids.astype(bool, order="F"),
        dir_scheme.offsets.astype(np.int32, order="F"),
        dir_scheme.codes.astype(np.uint8, order="F"),
    )
    raise_fortran_error("compute_dist2sink", err_code)
    return dists.astype(np.float32, order="F")
