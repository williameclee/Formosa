# Last modified
#   2026-02-11, En-Chi Lee (williameclee@arizona.edu)
#     - Rename flowdir functions to be more descriptive
#   2026-06-09, En-Chi Lee (williameclee@gmail.com)
#     - Added wrapper function for ridge distance computation in `DEMGrid` class
#     - Removed Numpy type `np.bool` to either `np.bool_` or `bool` for compatibility with newer Numpy versions
#   2026-06-11, En-Chi Lee (williameclee@gmail.com)
#     - Updated function and argument names to match the standardised names
#   2026-06-30, En-Chi Lee (williameclee@gmail.com)
#     - Added aliases to properties
#     - Added properties `ridgedir` and `ridge_strahler_order`
#   2026-07-01, En-Chi Lee (williameclee@gmail.com)
#     - Added property `shape`
#   2026-08-04, En-Chi Lee (williameclee@gmail.com)
#     - Updated `compute_dist2conf_max` and related functions' interface to reflect FORTRAN backend changes
#   2026-08-06, En-Chi Lee (williameclee@gmail.com)
#     - Added method `invalidate_ocean_basins` and replaced old `detect_ocean_basin`

from pathlib import Path
import warnings
import numpy as np
import rasterio
import rasterio.transform as rt
import scipy.ndimage as ndi

from formosa.dem.demio import read_dem
from formosa.geomorphology import D8Directions
from formosa.geomorphology import (
    get_neighbour_values,
    compute_slope,
    fill_depressions,
    invalidate_ocean_basins as _invalidate_ocean_basins,
    compute_flowdir,
    create_flowgraph,
    count_indegree,
    compute_flow_accumulation,
    compute_flow_strahler_order,
    compute_dist2source,
    compute_dist2sink,
    compute_dist2conf_max,
    compute_ridgedir,
    compute_dist2ridge,
    compute_ridge_strahler_order,
    label_watersheds,
)

from typing import Optional
import numpy.typing as npt


class DEMGrid:
    _original_dem: npt.NDArray[np.number]
    dem: npt.NDArray[np.number]
    x: npt.NDArray[np.floating | np.integer]
    y: npt.NDArray[np.floating | np.integer]
    transform: rasterio.Affine
    i: npt.NDArray[np.uint32]
    j: npt.NDArray[np.uint32]
    valid: npt.NDArray[np.bool_]

    def __init__(
        self,
        dem: npt.NDArray[np.number] | str | Path,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        transform: rasterio.Affine | None = None,
        gaussian_filter: float | None = None,
        stride: int | None = None,
        detect_ocean: bool | float | int = False,
        directions: D8Directions = D8Directions(),
        astype: type | np.dtype | None = None,
        min_ocean_size: int = 1,
        ocean_flood_below: bool = True,
    ):
        if isinstance(dem, (str, Path)):
            # Read a supported raster DEM (including GeoTIFF and SRTM HGT).
            dem, x, y, transform = read_dem(dem)
            self._original_dem = dem
            self.dem = dem
            self.x = x
            self.y = y
            self.transform = transform
        elif isinstance(dem, np.ndarray):
            self._original_dem = dem
            self.dem = dem

            self.transform = (
                transform if transform is not None else rasterio.Affine.identity()
            )

            # Generate x and y coordinates if not provided
            if x is None or y is None:
                ii, jj = np.meshgrid(
                    np.arange(self.dem.shape[1]), np.arange(self.dem.shape[0])
                )  # x and y indices
                self.x, self.y = rt.xy(transform, jj, ii)  # x and y coordinates
                self.x = np.reshape(self.x, (-1,)).reshape(self.dem.shape)
                self.y = np.reshape(self.y, (-1,)).reshape(self.dem.shape)
            else:
                assert (
                    x.shape == dem.shape and y.shape == dem.shape
                ), f"Provided x and y coordinates must match the shape of the DEM array (got DEM: {dem.shape}, x: {x.shape}, y: {y.shape})"
                self.x = x
                self.y = y
        else:
            raise TypeError(
                f"DEM must be either a file path or a numpy ndarray, got {type(dem)} instead."
            )

        if stride is not None:
            assert (
                stride > 0
            ), f"STRIDE must be a positive integer, got {stride} instead"

            self.stride = stride
            self.transform = rasterio.Affine(
                self.transform.a * stride,
                self.transform.b * stride,
                self.transform.c,
                self.transform.d * stride,
                self.transform.e * stride,
                self.transform.f,
            )

            self.dem = self.dem[::stride, ::stride]
            self.x = self.x[::stride, ::stride]
            self.y = self.y[::stride, ::stride]
        else:
            self.stride = 1

        if xlim is not None:
            if xlim[0] > xlim[1]:
                warnings.warn(
                    f"X limits are inverted: {xlim}. Swapping the limits.",
                    UserWarning,
                )
                xlim = (xlim[1], xlim[0])
            if xlim[0] < self.x.min() or xlim[1] > self.x.max():
                warnings.warn(
                    f"X limits {xlim} are out of bounds ({self.x.min()}, {self.x.max()}). Clipping the limits.",
                    UserWarning,
                )
                xlim = (max(xlim[0], self.x.min()), min(xlim[1], self.x.max()))

            orig_minx = self.x.min()
            x_mask = (self.x >= xlim[0]) & (self.x <= xlim[1])
            # mask with nan
            self.dem[~x_mask] = np.nan
            # drop all nan columns
            is_nan_row = np.all(np.isnan(self.dem), axis=1)
            is_nan_column = np.all(np.isnan(self.dem), axis=0)
            self.dem = self.dem[~is_nan_row, :][:, ~is_nan_column]
            self.x = self.x[~is_nan_row, :][:, ~is_nan_column]
            self.y = self.y[~is_nan_row, :][:, ~is_nan_column]

            new_minx = self.x.min()
            self.transform = rasterio.Affine(
                self.transform.a,
                self.transform.b,
                new_minx,
                self.transform.d,
                self.transform.e,
                self.transform.f,
            )

        if ylim is not None:
            if ylim[0] > ylim[1]:
                warnings.warn(
                    f"Y limits are inverted: {ylim}. Swapping the limits.",
                    UserWarning,
                )
                ylim = (ylim[1], ylim[0])
            if ylim[0] < self.y.min() or ylim[1] > self.y.max():
                warnings.warn(
                    f"Y limits {ylim} are out of bounds ({self.y.min()}, {self.y.max()}). Clipping the limits.",
                    UserWarning,
                )
                ylim = (max(ylim[0], self.y.min()), min(ylim[1], self.y.max()))

            orig_miny = self.y.min()

            y_mask = (self.y >= ylim[0]) & (self.y <= ylim[1])
            # mask with nan
            self.dem[~y_mask] = np.nan
            # drop all nan rows
            is_nan_row = np.all(np.isnan(self.dem), axis=1)
            is_nan_column = np.all(np.isnan(self.dem), axis=0)
            self.dem = self.dem[~is_nan_row, :][:, ~is_nan_column]
            self.x = self.x[~is_nan_row, :][:, ~is_nan_column]
            self.y = self.y[~is_nan_row, :][:, ~is_nan_column]

            new_miny = self.y.min()
            self.transform = rasterio.Affine(
                self.transform.a,
                self.transform.b,
                self.transform.c,
                self.transform.d,
                self.transform.e,
                new_miny,
            )

        if astype is not None:
            self.dem = self.dem.astype(astype)

        if self.dem.ndim != 2:
            raise ValueError(
                f"DEM must be a 2D array, got {self.dem.ndim}D array ({self.dem.shape}) instead."
            )
        self.i, self.j = np.meshgrid(
            np.arange(self.dem.shape[0]).astype(np.uint32),
            np.arange(self.dem.shape[1]).astype(np.uint32),
            indexing="ij",
        )

        self.valid = ~np.isnan(self.dem)
        self._ocean_mask = None
        self.ocean_threshold = None
        self._min_ocean_size = min_ocean_size
        self._ocean_flood_below = ocean_flood_below
        self.directions = directions
        ocean_detection_enabled = (
            bool(detect_ocean) if isinstance(detect_ocean, (bool, np.bool_)) else True
        )
        if ocean_detection_enabled:
            self.ocean_threshold = (
                0 if isinstance(detect_ocean, (bool, np.bool_)) else detect_ocean
            )
            previous_valid = self.valid.copy()
            self.valid = _invalidate_ocean_basins(
                self.dem,
                valids=self.valid,
                ocean_level=self.ocean_threshold,
                flood_below=self._ocean_flood_below,
                min_size=self._min_ocean_size,
                dir_scheme=self.directions,
            )
            self._ocean_mask = previous_valid & ~self.valid

        self.gaussian_filter = gaussian_filter
        if gaussian_filter is not None:
            filtered_dem = ndi.gaussian_filter(self.dem, sigma=gaussian_filter)
            self.dem = np.where(self.valid, filtered_dem, self.dem)

        self.quality = np.zeros(self.dem.shape, dtype=np.int16)
        self._slope: None | npt.NDArray[np.integer | np.floating] = None
        self._flat: None | npt.NDArray[np.bool_] = None
        self._flat_gradient: None | npt.NDArray[np.integer] = None
        self._flowdir: None | npt.NDArray[np.integer] = None
        self._indegree: None | npt.NDArray[np.integer] = None
        self._accumulation: None | npt.NDArray[np.integer | np.floating] = None
        self._strahler_order: None | npt.NDArray[np.uint8] = None
        self._watershed: None | npt.NDArray[np.int32] = None
        self._graphx = None
        self._graphy = None
        self._flowdist: None | npt.NDArray[np.floating] = None
        self._backdist: None | npt.NDArray[np.floating] = None
        self._bmax: None | npt.NDArray[np.floating] = None
        self._ridgedir: None | npt.NDArray[np.uint8] = None
        self._ridge_strahler_order: None | npt.NDArray[np.uint8] = None
        self._ridge_dist: None | npt.NDArray[np.float32] = None

    @property
    def shape(self) -> tuple[int]:
        return self.dem.shape

    @property
    def slope(self) -> npt.NDArray[np.floating | np.integer]:
        if self._slope is not None:
            return self._slope

        self._slope = compute_slope(self.dem, x=self.x, y=self.y)
        self._slope[~self.valid] = np.nan
        return self._slope

    @property
    def ocean_mask(self) -> npt.NDArray[np.bool_]:
        """
        Boolean mask representing cells connected to a sufficiently
        large ocean that touches the DEM edge.

        The elevation at or at and below which is controlled by the
        `ocean_threshold` property, and the minimum ocean size (in
        number of cells) is controlled by the private
        `min_ocean_size` property that can be set during
        initialisation.

        Notes
        -----
        See :func:`invalidate_ocean_basins` for more details.
        """

        if self._ocean_mask is None or self.ocean_threshold is None:
            if self.ocean_threshold is None:
                self.ocean_threshold = 0
            self.invalidate_ocean_basins(
                ocean_level=self.ocean_threshold,
                min_size=self._min_ocean_size,
                flood_below=self._ocean_flood_below,
            )
        assert self._ocean_mask is not None
        return self._ocean_mask

    @property
    def sea_mask(self) -> npt.NDArray[np.bool_]:
        """
        Boolean mask representing cells connected to a sufficiently
        large ocean that touches the DEM edge.

        Notes
        -----
        This is an alias of the property :func:`ocean_mask`.
        See :func:`invalidate_ocean_basins` for more details.
        """
        return self.ocean_mask

    @property
    def flowdir(self) -> npt.NDArray[np.integer]:
        if self._flowdir is None:
            self._flowdir, self._flat, self._flat_gradient = compute_flowdir(
                self.dem,
                dir_scheme=self.directions,
                valids=self.valid,
                resolve_flat=True,
            )
        return self._flowdir

    def flowdir_graph_xy(
        self,
        valid: npt.NDArray[np.bool_] | None = None,
    ) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]]:
        graphy, graphx = create_flowgraph(
            self.flowdir,
            dir_scheme=self.directions,
            valids=valid if valid is not None else self.valid,
            x=self.x.astype(np.float64),
            y=self.y,
        )
        return graphx, graphy

    @property
    def indegree(self) -> npt.NDArray[np.integer]:
        if self._indegree is None:
            self._indegree = count_indegree(self.flowdir, dir_scheme=self.directions)
        return self._indegree

    @property
    def accumulation(self) -> np.ndarray:
        if self._accumulation is None:
            self._accumulation = compute_flow_accumulation(
                self.flowdir,
                valids=self.valid,
                indegs=self.indegree,
                dir_scheme=self.directions,
            )
        return self._accumulation

    @property
    def strahler_order(self) -> npt.NDArray[np.uint8]:
        if self._strahler_order is None:
            self._strahler_order = compute_flow_strahler_order(
                self.flowdir,
                dir_scheme=self.directions,
            )
        return self._strahler_order

    def fill_depressions(self, max_fill_size: Optional[int] = None) -> "DEMGrid":
        """
        Fill enclosed depressions in-place using priority-flood.

        Parameters
        ----------
        max_fill_size : int, optional
            Maximum size (in cells) of a depression before it is
            considered an internally-drained basin instead.
            If `None`, all depressions are filled (equivalent to
            infinity).
            Default size is `None`.

        Returns
        -------
        DEM : DEMGrid
            DEM with depressions filled.

        Notes
        -----
        This is a wrapper for the function :func:`fill_depressions`.
        """
        self.dem = fill_depressions(
            self.dem, valids=self.valid, max_fill_size=max_fill_size
        )
        return self

    def invalidate_ocean_basins(
        self,
        ocean_level: int | float = 0,
        min_size: int = 1,
        flood_below: bool = True,
    ) -> "DEMGrid":
        """
        Marks sufficiently large boundary-connected ocean basins as
        invalid, which means they would not participate in
        calculation of flow directions, etc.

        Parameters
        ----------
        ocean_level : int | float, optional
            Elevation threshold defining ocean cells.
            Default elevation is `0`.
        min_size : int, optional
            Minimum cell count threshold for ocean basin
            invalidation.
            Ocean basins containing at least this number of cells
            are invalidated.
            Default size is `1`.
        flood_below : bool, optional
            Whether elevations strictly below `ocean_level` qualify
            as ocean cells.
            When false, only cells exactly equal to `ocean_level`
            qualify.
            Default option is `True`.

        Returns
        -------
        DEM : DEMGrid
            DEM with ocean basins marked as invalid.
            Internal information (e.g. ocean level) is also updated.

        Notes
        -----
        This is a wrapper for the function :func:`invalidate_ocean_basins`.
        """
        previous_valid = self.valid.copy()
        self.valid = _invalidate_ocean_basins(
            self.dem,
            valids=self.valid,
            ocean_level=ocean_level,
            flood_below=flood_below,
            min_size=min_size,
            dir_scheme=self.directions,
        )

        newly_invalid = previous_valid & ~self.valid
        if self._ocean_mask is None:
            self._ocean_mask = newly_invalid
        else:
            self._ocean_mask |= newly_invalid
        self.ocean_threshold = ocean_level
        self._min_ocean_size = min_size
        self._ocean_flood_below = flood_below

        # Cached terrain products may depend directly or indirectly
        # on valid
        self._slope = None
        self._flat = None
        self._flat_gradient = None
        self._flowdir = None
        self._indegree = None
        self._accumulation = None
        self._strahler_order = None
        self._watershed = None
        self._graphx = None
        self._graphy = None
        self._flowdist = None
        self._backdist = None
        self._bmax = None
        self._ridgedir = None
        self._ridge_strahler_order = None
        self._ridge_dist = None
        return self

    @property
    def dist2source(self) -> npt.NDArray[np.floating]:
        if self._flowdist is None:
            self._flowdist = compute_dist2source(
                self.flowdir,
                dir_scheme=self.directions,
                x=self.x,
                y=self.y,
                valids=self.valid,
                indegs=self.indegree,
            )
        return self._flowdist

    @property
    def flow_distance(self) -> npt.NDArray[np.floating]:
        return self.dist2source

    @property
    def watersheds(self) -> npt.NDArray[np.int32]:
        if self._watershed is not None:
            return self._watershed

        self._watershed = label_watersheds(
            self.flowdir,
            dir_scheme=self.directions,
            valids=self.valid,
        )
        return self._watershed

    @property
    def dist2sink(self) -> npt.NDArray[np.floating]:
        if self._backdist is not None:
            return self._backdist

        self._backdist = compute_dist2sink(
            self.flowdir,
            dir_scheme=self.directions,
            x=self.x,
            y=self.y,
            valids=self.valid,
        )
        return self._backdist

    @property
    def backdist(self) -> npt.NDArray[np.floating]:
        return self.dist2sink

    @property
    def bmax(self) -> npt.NDArray[np.floating]:
        if self._bmax is not None:
            return self._bmax

        self._bmax = compute_dist2conf_max(
            self.flowdir.astype(np.uint8, order="F"),
            self.valid.astype(np.bool_, order="F"),
            self.x.astype(np.float32, order="F"),
            self.y.astype(np.float32, order="F"),
            dir_scheme=self.directions,
        )
        return self._bmax

    @property
    def ridge_dist(self) -> npt.NDArray[np.floating]:
        """
        'Distance' to the ridge, approximated by the distance to sink in the maximum confluence distance landscape.

        Returns
        -------
        dist : npt.NDArray[np.float32]
            Distance to the ridge, approximated by the distance to sink in the maximum confluence distance landscape.
        """
        return self.dist2ridge

    @property
    def ridgedir(self) -> npt.NDArray[np.uint8]:
        if self._ridgedir is not None:
            return self._ridgedir
        self._ridgedir = compute_ridgedir(
            self.flowdir,
            dir_scheme=self.directions,
            valids=self.valid,
            x=self.x,
            y=self.y,
        )
        return self._ridgedir

    @property
    def dist2ridge(self) -> npt.NDArray[np.floating]:
        """
        'Distance' to the ridge, approximated by the distance to sink in the maximum confluence distance landscape.

        Returns
        -------
        dist : npt.NDArray[np.float32]
            Distance to the ridge, approximated by the distance to sink in the maximum confluence distance landscape.
        """

        if self._ridge_dist is not None:
            return self._ridge_dist

        self._ridge_dist = compute_dist2ridge(
            self.ridgedir,
            valids=self.valid.astype(np.bool_, order="F"),
            x=self.x.astype(np.float32, order="F"),
            y=self.y.astype(np.float32, order="F"),
            dir_scheme=self.directions,
            dir_is_ridge=True,
        )
        return self._ridge_dist

    @property
    def ridge_strahler_order(self) -> npt.NDArray[np.uint8]:
        if self._ridge_strahler_order is not None:
            return self._ridge_strahler_order
        self._ridge_strahler_order = compute_ridge_strahler_order(
            self.ridgedir,
            dir_scheme=self.directions,
            valids=self.valid,
            dir_is_ridge=True,
        )
        return self._ridge_strahler_order


def fill_pits(
    dem: npt.NDArray[np.number],
) -> tuple[npt.NDArray[np.number], npt.NDArray[np.bool_]]:
    dem_filled = dem.copy()

    min_neighbours = np.min(get_neighbour_values(dem_filled)[0], axis=0)
    is_pit = dem_filled < min_neighbours
    dem_filled[is_pit] = min_neighbours[is_pit]

    return dem_filled, is_pit
