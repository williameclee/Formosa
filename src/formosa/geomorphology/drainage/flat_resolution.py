"""
Resolves flat areas in digital elevation models for flow routing.

The algorithms assign synthetic gradients to flats and mainly follow
Barnes *et al.* (2014), https://doi.org/10.1016/j.cageo.2013.01.009.

Last modified: 2026-08-23, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np
from numpy.typing import NDArray

import formosa.geomorphology.drainage._backends.flat_resolution_py as fres_py
from formosa.geomorphology._native import drainage_flat_resolution as flat_f
from formosa.geomorphology.drainage.directions import D8Directions
from formosa.geomorphology.drainage.neighbours import get_neighbour_values
from formosa.geomorphology.raster_validation import (
    validate_format_dem,
    validate_format_dir_scheme,
    validate_format_flowdirs,
    validate_format_valids,
)
from formosa.utils import Backend, NpFlowDir, NpReal, raise_fortran_error
from formosa.utils.validation import validate_same_shape


def find_flat_edges(
    dem: NDArray[NpReal],
    dirs: NDArray[NpFlowDir],
    dir_scheme: D8Directions | None = None,
    valids: NDArray[np.bool_] | None = None,
    backend: Backend = "fortran",
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """
    Finds the cells on the edges of flat areas that drain to lower terrain (low edges) and those that are adjacent to higher terrain (high edges).
    From [R. Barnes *et al.* (2014)](https://doi.org/10.1016/j.cageo.2013.01.009), Algorithm 3 (p. 133).

    Parameters
    ----------
    dem : NDArray[number]
        Digital elevation model raster.
        - Expected shape: `(nrows, ncols)`.
    dirs : NDArray[uint8]
        Flow direction raster.
        - Expected shape: `(nrows, ncols)`, same as `dem`.
    dir_scheme : D8Directions, optional
        Instance of `D8Directions` defining the flow direction
        scheme.
        - Default scheme is `D8Directions()`.
    valids : NDArray[bool], optional
        Boolean mask indicating valid cells in the DEM.
        If `None`, all cells are considered valid.
        - Expected shape: `(nrows, ncols)`, same as `dem`.
        - Default mask is `None`.
    backend : {'fortran', 'python'}, optional
        Backend to use for computation.
        `'fortran'` uses the Fortran extension for performance,
        while `'python'` uses a pure Python implementation.
        - Default backend is `'fortran'`.

    Returns
    -------
    low_edges : NDArray[bool]
        Boolean mask indicating low-edge cells of flat areas.
        - Shape: `(nrows, ncols)`, same as `dem`.
    high_edges : NDArray[bool]
        Boolean mask indicating high-edge cells of flat areas.
        - Shape: `(nrows, ncols)`, same as `dem`.
    """
    dem = validate_format_dem(dem)
    dirs = validate_format_flowdirs(dirs, dem)
    dir_scheme = validate_format_dir_scheme(dir_scheme)
    valids = validate_format_valids(valids, dem)
    match backend:
        case "python":
            low_edges, high_edges = fres_py.find_flat_edges(
                dem, dirs, dir_scheme=dir_scheme
            )
        case "fortran":
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
    dem: NDArray[NpReal],
    seeds: NDArray[np.bool_],
    dir_scheme: D8Directions | None = None,
    valids: NDArray[np.bool_] | None = None,
) -> NDArray[np.int32]:
    """
    Separates and labels inidividual flat areas in a DEM.
    From [R. Barnes *et al.* (2014)](https://doi.org/10.1016/j.cageo.2013.01.009), Algorithm 4 (p. 133).

    Parameters
    ----------
    dem : NDArray[number]
        Digital elevation model raster.
        - Expected shape: `(nrows, ncols)`.
    seeds : NDArray[bool]
        Boolean mask indicating flat area locations.
        - Expected shape: `(nrows, ncols)`, same as `dem`.
    dir_scheme : D8Directions, optional
        Instance of `D8Directions` defining the flow direction
        scheme.
        - Default scheme is `D8Directions()`.
    valids : NDArray[bool], optional
        Boolean mask indicating valid cells in the DEM.
        If `None`, all cells are considered valid.
        - Expected shape: `(nrows, ncols)`, same as `dem`.
        - Default mask is `None`.

    Returns
    -------

    Raises
    ------
    TypeError
        If the input seeds is not of the expected type or format.
    ValueError
        If the shapes of the input arrays do not match the expected dimensions.
    labels : NDArray[int32]
        Unique integer label for each flat region.
        - Shape: `(nrows, ncols)`, same as `dem`.
    """
    dem = validate_format_dem(dem)
    validate_same_shape(seeds, dem, "seed mask", "DEM")
    dir_scheme = validate_format_dir_scheme(dir_scheme)
    valids = validate_format_valids(valids, dem, "DEM")

    labels, err_code = flat_f.label_flats(
        dem.astype(np.float32, order="F"),
        seeds.astype(bool, order="F"),
        valids.astype(bool, order="F"),
        dir_scheme.offsets.astype(np.int32, order="F"),
    )
    raise_fortran_error("label_flats", err_code)

    return labels.astype(np.int32, order="F")


def find_flat(
    dem: NDArray[NpReal],
    dir_scheme: D8Directions | None = None,
    valids: NDArray[np.bool_] | None = None,
    only_min: bool = True,
) -> NDArray[np.bool_]:
    """
    Identifies flat areas in a DEM where cells have no lower
    neighbouring cells.

    Parameters
    ----------
    dem : NDArray[number]
        Digital elevation model raster.
        - Expected shape: `(nrows, ncols)`.
    dir_scheme : D8Directions, optional
        Instance of `D8Directions` defining the neighbourhood for
        the synthetic elevation gradient.
        - Default scheme is `D8Directions()`.
    valids : NDArray[bool], optional
        Boolean mask indicating valid cells in the DEM.
        If `None`, all cells are considered valid.
        - Expected shape: `(nrows, ncols)`, same as `dem`.
        - Default mask is `None`.
    only_min : bool, optional
        Whether only cells strictly equal to the minimum of their
        neighbours qualify as flat.
        If False, cells equal to any neighbour are considered flat.
        - Default option is `True`.

    Returns
    -------
    flats : NDArray[bool]
        Boolean mask indicating cells belonging to flat areas.
        - Shape: `(nrows, ncols)`, same as `dem`.
    """
    dem = validate_format_dem(dem)
    dir_scheme = validate_format_dir_scheme(dir_scheme)
    valids = validate_format_valids(valids, dem, "DEM")
    if np.any(~valids):
        dem[~valids] = np.max(dem[~valids]) + 1

    neighbours, _, _ = get_neighbour_values(
        dem, dir_scheme=dir_scheme, pad_val=np.nan, include_self=False
    )
    if only_min:
        flats = dem == np.nanmin(neighbours, axis=0)
    else:
        flats = np.any(dem == neighbours, axis=0)

    flats = flats & valids
    return flats


def find_ambiguous(
    dem: NDArray[NpReal],
    dir_scheme: D8Directions | None = None,
) -> NDArray[np.bool_]:
    """
    Detects ambiguous flow directions in a DEM, where multiple neighbouring cells have the same minimum elevation.

    Parameters
    ----------
    dem : NDArray[number]
        Digital elevation model raster.
        - Expected shape: `(nrows, ncols)`.
    dir_scheme : D8Directions, optional
        Instance of `D8Directions` defining the flow direction
        scheme.
        - Default scheme is `D8Directions()`.

    Returns
    -------
    ambiguities : NDArray[bool]
        Boolean mask indicating cells with ambiguous flow
        directions.
        - Shape: `(nrows, ncols)`, same as `dem`.
    """
    dem = validate_format_dem(dem)
    dir_scheme = validate_format_dir_scheme(dir_scheme)

    nabrs, _, _ = get_neighbour_values(dem, dir_scheme=dir_scheme)
    min_nabrs = np.min(nabrs, axis=0)
    ambiguities = np.sum(nabrs == min_nabrs, axis=0) > 1
    ambiguities = ambiguities & ~(find_flat(dem))
    return ambiguities


def create_pushing_syn_grad(
    labels: NDArray[np.integer],
    high_edges: NDArray[np.bool_],
    dir_scheme: D8Directions = D8Directions(),
) -> NDArray[np.int32]:
    """
    Produces a synthetic elevation that decreases away from 'high edges' of flats.
    Modified from [R. Barnes *et al.* (2014)](https://doi.org/10.1016/j.cageo.2013.01.009), Algorithm 5 (p. 133–134).

    Parameters
    ----------
    labels : NDArray[number]
        Integer label raster for flat regions.
        Non-flat areas are labelled with `0`, and flat areas have
        positive integer labels starting from `1`.
        - Expected shape: `(nrows, ncols)`.
    high_edges : NDArray[bool]
        Boolean mask indicating high-edge locations.
        - Expected shape: `(nrows, ncols)`, same as `labels`.
    dir_scheme : D8Directions, optional
        Instance of `D8Directions` defining neighbour offsets.
        - Default scheme is `D8Directions()`.

    Returns
    -------
    z_syn : NDArray[int32]
        Synthetic elevation increasing away from high edges within
        each flat region.
        - Shape: `(nrows, ncols)`, same as `labels`.
    """
    validate_same_shape(labels, high_edges, "label raster", "high edge mask")

    z_syn, err_code = flat_f.create_pushing_syn_grad(
        labels.astype(np.int32, order="F"),
        high_edges.astype(bool, order="F"),
        dir_scheme.offsets.astype(np.int32, order="F"),
    )
    raise_fortran_error("create_pushing_syn_grad", err_code)
    return z_syn.astype(np.int32, order="F")


def create_pulling_syn_grad(
    labels: NDArray[np.integer],
    low_edges: NDArray[np.bool_],
    dir_scheme: D8Directions = D8Directions(),
) -> NDArray[np.integer]:
    """
    Produces a synthetic elevation that drains towards 'low edges' of flats.
    Modified from [R. Barnes *et al.* (2014)](https://doi.org/10.1016/j.cageo.2013.01.009), Algorithm 6 (p. 134).

    Parameters
    ----------
    labels : NDArray[number]
        Integer label raster for flat regions.
        Non-flat areas are labelled with `0`, and flat areas have
        positive integer labels starting from `1`.
        - Expected shape: `(nrows, ncols)`.
    low_edges : NDArray[bool]
        Boolean mask indicating low-edge locations.
        - Expected shape: `(nrows, ncols)`, same as `labels`.
    dir_scheme : D8Directions, optional
        An instance of D8Directions defining the flow direction scheme, here it is used to determine the offsets for neighbor cells.
        Default is `D8Directions()`.

    Returns
    -------
    z_syn : NDArray[integer]

    Raises
    ------
    TypeError
        If the input low_edges is not of the expected type or format.
    ValueError
        If the shapes of the input arrays do not match the expected dimensions.
        Synthetic elevation increasing towards low edges within each
        flat region.
        - Shape: `(nrows, ncols)`, same as `labels`.
    """
    validate_same_shape(labels, low_edges, "flat label raster", "low edges mask")
    
    z_syn, err_code = flat_f.create_pulling_syn_grad(
        labels.astype(np.int32, order="F"),
        low_edges.astype(bool, order="F"),
        dir_scheme.offsets.astype(np.int32, order="F"),
    )
    raise_fortran_error("create_pulling_syn_grad", err_code)
    return z_syn


def compute_syn_flowdir(
    z: NDArray[NpReal],
    labels: NDArray[np.integer],
    dir_scheme: D8Directions = D8Directions(),
    backend: Backend = "fortran",
) -> NDArray[np.uint8]:
    """
    Computes flow directions within flat areas using synthetic elevation.
    Very similar to the naive flow direction computation, but only search within the same flat area.

    Parameters
    ----------
    z : NDArray[number]
        Synthetic elevation raster within flat areas.
        - Expected shape: `(nrows, ncols)`.
    labels : NDArray[int]
        Integer label raster for flat regions.
        - Expected shape: `(nrows, ncols)`, same as `z`.
    dir_scheme : D8Directions, optional
        Instance of `D8Directions` defining the flow direction scheme.
        - Default scheme is `D8Directions()`.
    backend : {'fortran', 'python'}, optional
        Backend to use for computation.
        `'fortran'` uses the Fortran extension for performance,
        while `'python'` uses a pure Python implementation.
        - Default backend is `'fortran'`.

    Returns
    -------
    dirs : NDArray[uint8]
        Flow directions within flat areas.
        - Shape: `(nrows, ncols)`, same as `z`.
    """
    z = validate_format_dem(z)
    validate_same_shape(labels, z, "label", "synthetic elevation rasters")
    match backend:
        case "python":
            dirs = fres_py.compute_masked_flowdir(z, labels, dir_scheme=dir_scheme)
        case "fortran":
            dirs = flat_f.compute_syn_flowdir(
                z.astype(np.int32, order="F"),
                labels.astype(np.int32, order="F"),
                dir_scheme.offsets.astype(np.int32, order="F"),
                dir_scheme.codes.astype(np.uint8, order="F"),
            )

    return dirs.astype(np.uint8, order="F")
