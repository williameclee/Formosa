# Last modified
#   2026-06-09, En-Chi Lee (williameclee@gmail.com)
#     - Renamed FORTRAN function call: `compute_masked_flowdir` ->
#       `compute_synthetic_flowdir`.
#     - Added `valids` argument to `label_flats` function.


import numpy as np

from formosa.utils import Backend, raise_fortran_error
from formosa.geomorphology.drainage.directions import D8Directions
from formosa.geomorphology.drainage.neighbours import (
    get_neighbour_values,
)
from formosa.geomorphology._native import drainage_flat_resolution as flat_f
import formosa.geomorphology.drainage._backends.flowdir_py as flowdir_py

from typing import Optional
import numpy.typing as npt


def find_flat_edges(
    dem: npt.NDArray[np.number],
    dirs: npt.NDArray[np.integer],
    dir_scheme: D8Directions = D8Directions(),
    valids: Optional[npt.NDArray[np.bool_]] = None,
    backend: Backend = "fortran",
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """
    Finds the cells on the edges of flat areas that drain to lower terrain (low edges) and those that are adjacent to higher terrain (high edges).
    From [R. Barnes *et al.* (2014)](https://doi.org/10.1016/j.cageo.2013.01.009), Algorithm 3 (p. 133).

    Parameters
    ----------
    dem : NDArray[number]
        A 2D array representing the digital elevation model (DEM).
    dirs : NDArray[integer]
        A 2D array representing the flow direction for each cell in the DEM.
    dir_scheme : D8Directions, optional
        An instance of D8Directions defining the flow direction scheme that `flowdirs` uses.
        Default is `D8Directions()`.
    valids : NDArray[bool], optional
        A boolean mask array indicating valid cells in the DEM.
        If `None`, all cells are considered valid.
        Default is `None`.
    backend : {'fortran', 'python'}, optional
        Backend to use for computation.
        `'fortran'` uses the FORTRAN extension for performance,
        while `'python'` uses a pure Python implementation.
        Default backend is `'fortran'`.

    Returns
    -------
    low_edges : NDArray[bool]
        A boolean mask array where True indicates cells that are low edges of flat areas.
    high_edges : NDArray[bool]
        A boolean mask array where True indicates cells that are high edges of flat areas.
    """
    match backend:
        case "python":
            low_edges, high_edges = flowdir_py.find_flat_edges(
                dem, dirs, dir_scheme=dir_scheme
            )
        case "fortran":
            if valids is None:
                valids = np.ones(dem.shape, dtype=bool, order="F")

            low_edges, high_edges = flat_f.find_flat_edges(
                dem.astype(np.float32, order="F"),
                dirs.astype(np.int32, order="F"),
                valids.astype(bool, order="F"),
                dir_scheme.offsets.astype(np.int32, order="F"),
                dir_scheme.codes.astype(np.uint8, order="F"),
            )

    return (
        low_edges.astype(bool, order="F"),
        high_edges.astype(bool, order="F"),
    )


def label_flats(
    dem: npt.NDArray[np.number],
    seeds: npt.NDArray[np.bool_],
    valids: Optional[npt.NDArray[np.bool_]] = None,
    dir_scheme: D8Directions = D8Directions(),
) -> npt.NDArray[np.int32]:
    """
    Separates and labels inidividual flat areas in a DEM.
    From [R. Barnes *et al.* (2014)](https://doi.org/10.1016/j.cageo.2013.01.009), Algorithm 4 (p. 133).

    Parameters
    ----------
    dem : NDArray[number]
        A 2D array representing the digital elevation model (DEM).
    seeds : NDArray[bool]
        Either a boolean mask array indicating flat area locations, or a 2D integer array of coordinates, or an iterable of coordinate pairs.
    valids : NDArray[bool], optional
        A boolean mask array indicating valid cells in the DEM.
        If `None`, all cells are considered valid.
        Default is `None`.
    dir_scheme : D8Directions, optional
        An instance of `D8Directions` defining the flow direction scheme.
        Default is `D8Directions()`.

    Returns
    -------
    labels : NDArray[int]
        A 2D integer array where each flat region is labeled with a unique integer.

    Raises
    ------
    TypeError
        If the input seeds is not of the expected type or format.
    ValueError
        If the shapes of the input arrays do not match the expected dimensions.
    """
    assert (
        dem.shape == seeds.shape
    ), f"Shapes for dem ({dem.shape}) and seeds ({seeds.shape}) do not match."
    if valids is not None:
        assert (
            dem.shape == valids.shape
        ), f"Shapes for dem ({dem.shape}) and valids ({valids.shape}) do not match."
    else:
        valids = np.ones(dem.shape, dtype=bool, order="F")

    labels, err_code = flat_f.label_flats(
        dem.astype(np.float32, order="F"),
        seeds.astype(bool, order="F"),
        valids.astype(bool, order="F"),
        dir_scheme.offsets.astype(np.int32, order="F"),
    )
    raise_fortran_error("label_flats", err_code)

    return labels.astype(np.int32, order="F")


def find_flat(
    dem: npt.NDArray[np.number],
    valids: Optional[npt.NDArray[np.bool_]] = None,
    only_min: bool = True,
    dir_scheme: D8Directions = D8Directions(window=3),
) -> npt.NDArray[np.bool_]:
    """
    Identifies flat areas in a DEM where cells have no lower neighbouring cells.

    Parameters
    ----------
    dem : NDArray[number]
        A 2D array representing the digital elevation model (DEM).
    valid : NDArray[bool], optional
        A boolean mask array indicating valid cells in the DEM.
        If `None`, all cells are considered valid.
        Default is `None`.
    only_min : bool, optional
        If True, only cells that are equal to the minimum of their neighbours are considered flat.
        If False, cells equal to any neighbour are considered flat.
        Default is True.
    dir_scheme : D8Directions, optional
        An instance of D8Directions defining the neighbour offsets.
        Default is D8Directions(window=3).

    Returns
    -------
    flats : NDArray[bool]
        A boolean mask array where True indicates cells that are part of flat areas.
    """
    if valids is not None and np.any(~valids):
        dem[~valids] = np.max(dem[~valids]) + 1

    neighbours, _, _ = get_neighbour_values(
        dem, dir_scheme=dir_scheme, pad_value=np.nan, include_self=False
    )
    if only_min:
        flats = dem == np.nanmin(neighbours, axis=0)
    else:
        flats = np.any(dem == neighbours, axis=0)

    if valids is not None:
        flats = flats & valids
    return flats


def find_ambiguous(
    dem: npt.NDArray[np.number],
    dir_scheme: D8Directions = D8Directions(),
) -> npt.NDArray[np.bool_]:
    """
    Detects ambiguous flow directions in a DEM, where multiple neighbouring cells have the same minimum elevation.

    Parameters
    ----------
    dem : NDArray[number]
        A 2D array representing the digital elevation model (DEM).
    dir_scheme : D8Directions, optional
        An instance of `D8Directions` defining the flow direction scheme.
        Default is `D8Directions()`.

    Returns
    -------
    ambiguities : NDArray[bool]
        A boolean mask array where True indicates cells with ambiguous flow directions.
    """
    neighbours, _, _ = get_neighbour_values(dem, dir_scheme=dir_scheme)
    min_neighbours = np.min(neighbours, axis=0)
    ambiguities = np.sum(neighbours == min_neighbours, axis=0) > 1
    ambiguities = ambiguities & ~(find_flat(dem))
    return ambiguities


def create_pushing_syn_grad(
    labels: npt.NDArray[np.number],
    high_edges: npt.NDArray[np.bool_],
    dir_scheme: D8Directions = D8Directions(),
) -> npt.NDArray[np.int32]:
    """
    Produces a synthetic elevation that decreases away from 'high edges' of flats.
    Modified from [R. Barnes *et al.* (2014)](https://doi.org/10.1016/j.cageo.2013.01.009), Algorithm 5 (p. 133–134).

    Parameters
    ----------
    labels : NDArray[number]
        A 2D array where each flat region is labeled with a unique integer.
        It is assumed that non-flat areas are labeled with 0, and flat areas have positive integer labels starting from 1 (the Fortran extension relies on this).
    high_edges : NDArray[bool]
        A boolean mask array indicating high edge locations.
    dir_scheme : D8Directions, optional
        An instance of D8Directions defining the flow direction scheme, here it is used to determine the offsets for neighbor cells.
        Default is `D8Directions()`.

    Returns
    -------
    z_syn : NDArray[int32]
        A 2D integer array representing the synthetic elevation that increases away from high edges within each flat region.

    Raises
    ------
    TypeError
        If the input high_edges is not of the expected type or format.
    ValueError
        If the shapes of the input arrays do not match the expected dimensions.
    """
    assert (
        labels.shape == high_edges.shape
    ), f"Shapes for labels ({labels.shape}) and high_edges ({high_edges.shape}) do not match."

    z_syn, err_code = flat_f.create_pushing_syn_grad(
        labels.astype(np.int32, order="F"),
        high_edges.astype(bool, order="F"),
        dir_scheme.offsets.astype(np.int32, order="F"),
    )
    raise_fortran_error("create_pushing_syn_grad", err_code)
    return z_syn.astype(np.int32, order="F")


def create_pulling_syn_grad(
    labels: npt.NDArray[np.number],
    low_edges: npt.NDArray[np.bool_],
    dir_scheme: D8Directions = D8Directions(),
) -> npt.NDArray[np.integer]:
    """
    Produces a synthetic elevation that drains towards 'low edges' of flats.
    Modified from [R. Barnes *et al.* (2014)](https://doi.org/10.1016/j.cageo.2013.01.009), Algorithm 6 (p. 134).

    Parameters
    ----------
    labels : NDArray[number]
        A 2D array where each flat region is labeled with a unique integer.
        It is assumed that non-flat areas are labeled with 0, and flat areas have positive integer labels starting from 1 (the Fortran extension relies on this).
    low_edges : NDArray[bool]
        A boolean mask array indicating low edge locations.
    dir_scheme : D8Directions, optional
        An instance of D8Directions defining the flow direction scheme, here it is used to determine the offsets for neighbor cells.
        Default is `D8Directions()`.

    Returns
    -------
    z_syn : NDArray[integer]
        A 2D integer array representing the synthetic elevation that increases towards low edges within each flat region.

    Raises
    ------
    TypeError
        If the input low_edges is not of the expected type or format.
    ValueError
        If the shapes of the input arrays do not match the expected dimensions.
    """
    z_syn, err_code = flat_f.create_pulling_syn_grad(
        labels.astype(np.int32, order="F"),
        low_edges.astype(bool, order="F"),
        dir_scheme.offsets.astype(np.int32, order="F"),
    )
    raise_fortran_error("create_pulling_syn_grad", err_code)
    return z_syn


def compute_syn_flowdir(
    z: npt.NDArray[np.integer | np.floating],
    labels: npt.NDArray[np.integer],
    dir_scheme: D8Directions = D8Directions(),
    backend: Backend = "fortran",
) -> npt.NDArray[np.uint8]:
    """
    Computes flow directions within flat areas using synthetic elevation.
    Very similar to the naive flow direction computation, but only search within the same flat area.

    Parameters
    ----------
    z : NDArray[int | float]
        A 2D array representing the synthetic elevation within flat areas.
    labels : NDArray[int]
        A 2D array where each flat region is labeled with a unique integer.
    dir_scheme : D8Directions, optional
        An instance of `D8Directions` defining the flow direction scheme.
        Default is `D8Directions()`.
    backend : {'fortran', 'python'}, optional
        Backend to use for computation.
        `'fortran'` uses the FORTRAN extension for performance,
        while `'python'` uses a pure Python implementation.
        Default backend is `'fortran'`.

    Returns
    -------
    dirs : NDArray[int]
        A 2D integer array representing the flow directions within flat areas.
    """
    match backend:
        case "python":
            dirs = flowdir_py.compute_masked_flowdir(z, labels, dir_scheme=dir_scheme)
        case "fortran":
            dirs = flat_f.compute_syn_flowdir(
                z.astype(np.int32, order="F"),
                labels.astype(np.int32, order="F"),
                dir_scheme.offsets.astype(np.int32, order="F"),
                dir_scheme.codes.astype(np.uint8, order="F"),
            )

    return dirs.astype(np.uint8, order="F")
