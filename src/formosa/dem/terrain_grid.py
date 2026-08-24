"""
Represents and processes gridded digital elevation models.

This module provides :class:`DEMGrid`, which coordinates raster
input and geomorphological operations on a digital elevation model
(DEM).

Last modified: 2026-08-24, En-Chi Lee (williameclee@gmail.com)
"""

import warnings
from pathlib import Path

import numpy as np
import rasterio
import rasterio.transform as rt
import scipy.ndimage as ndi
from numpy.typing import NDArray

from formosa.dem.demio import read_dem
from formosa.geomorphology.drainage import (
    DirectionEncoding,
    compute_dist2conf_max,
    compute_dist2ridge,
    compute_dist2sink,
    compute_dist2source,
    compute_flow_accumulation,
    compute_flow_strahler_order,
    compute_flowdir,
    compute_ridge_strahler_order,
    compute_ridgedir,
    count_indegree,
    fill_depressions,
    get_neighbour_values,
    label_watersheds,
)
from formosa.geomorphology.drainage import (
    invalidate_ocean_basins as _invalidate_ocean_basins,
)
from formosa.geomorphology.drainage.network import create_flowline_plot_data
from formosa.geomorphology.raster_validation import validate_format_dir_encoding
from formosa.geomorphology.terrain import compute_prominence, compute_slope
from formosa.utils import NpCoords, NpReal, NpFlowDir, NpCanonIndex
from formosa.utils.validation import validate_same_shape


class DEMGrid:
    _original_dem: NDArray[NpReal]
    dem: NDArray[NpReal]
    x: NDArray[NpCoords]
    y: NDArray[NpCoords]
    transform: rasterio.Affine
    i: NDArray[np.uint32]
    j: NDArray[np.uint32]
    valid: NDArray[np.bool_]
    dir_enc: DirectionEncoding

    def __init__(
        self,
        dem: NDArray[NpReal] | str | Path,
        x: NDArray[NpCoords] | None = None,
        y: NDArray[NpCoords] | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        transform: rasterio.Affine | None = None,
        gaussian_filter: float | None = None,
        stride: int | None = None,
        detect_ocean: bool | float = False,
        dir_enc: DirectionEncoding | None = None,
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
                validate_same_shape(x, dem, "X coordinates", "DEM")
                validate_same_shape(y, dem, "Y coordinates", "DEM")
                self.x = x
                self.y = y
        else:
            raise TypeError(
                f"DEM must be either a file path or a numpy ndarray, got {type(dem)} instead."
            )

        if stride is not None:
            assert (
                stride > 0
            ), f"Stride must be a positive integer, got {stride} instead"

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
        self.dir_enc = validate_format_dir_encoding(dir_enc)
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
                self.dir_enc,
                valids=self.valid,
                ocean_lvl=self.ocean_threshold,
                flood_below=self._ocean_flood_below,
                min_size=self._min_ocean_size,
            )
            self._ocean_mask = previous_valid & ~self.valid

        self.gaussian_filter = gaussian_filter
        if gaussian_filter is not None:
            filtered_dem = ndi.gaussian_filter(self.dem, sigma=gaussian_filter)
            self.dem = np.where(self.valid, filtered_dem, self.dem)

        self.quality = np.zeros(self.dem.shape, dtype=np.int16)
        self._slope: None | NDArray[NpReal] = None
        self._flat: None | NDArray[np.bool_] = None
        self._flat_gradient: None | NDArray[np.integer] = None
        self._flowdir: None | NDArray[np.uint8] = None
        self._indegs: None | NDArray[np.int8] = None
        self._accums: None | NDArray[np.float64] = None
        self._strahler_order: None | NDArray[np.uint8] = None
        self._ws: None | NDArray[np.int32] = None
        self._graphx = None
        self._graphy = None
        self._flowdist: None | NDArray[np.floating] = None
        self._backdist: None | NDArray[np.floating] = None
        self._bmax: None | NDArray[np.floating] = None
        self._ridgedir: None | NDArray[np.uint8] = None
        self._ridge_strahler_order: None | NDArray[np.uint8] = None
        self._ridge_dist: None | NDArray[np.float32] = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.dem.shape

    @property
    def slope(self) -> NDArray[NpReal]:
        if self._slope is not None:
            return self._slope

        self._slope = compute_slope(self.dem, x=self.x, y=self.y)
        self._slope[~self.valid] = np.nan
        return self._slope

    @property
    def prominence(self) -> NDArray[NpReal]:
        proms, _, _, _, _, _ = compute_prominence(self.dem, self.dir_enc, self.valid)
        return proms

    @property
    def ocean_mask(self) -> NDArray[np.bool_]:
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
                ocean_lvl=self.ocean_threshold,
                min_size=self._min_ocean_size,
                flood_below=self._ocean_flood_below,
            )
        assert self._ocean_mask is not None
        return self._ocean_mask

    @property
    def sea_mask(self) -> NDArray[np.bool_]:
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
    def flowdir(self) -> NDArray[NpFlowDir]:
        if self._flowdir is None:
            self._flowdir, self._flat, self._flat_gradient = compute_flowdir(
                self.dem, self.dir_enc, valids=self.valid, resolve_flat=True
            )
        return self._flowdir

    def flowdir_graph_xy(
        self, valid: NDArray[np.bool_] | None = None
    ) -> tuple[NDArray[NpCanonIndex], NDArray[NpCanonIndex]]:
        graphy, graphx = create_flowline_plot_data(
            self.flowdir,
            self.dir_enc,
            valids=valid if valid is not None else self.valid,
            x=self.x.astype(np.float64),
            y=self.y.astype(np.float64),
        )
        return graphx, graphy

    @property
    def indegree(self) -> NDArray[np.int8]:
        if self._indegs is None:
            self._indegs = count_indegree(self.flowdir, self.dir_enc)
        return self._indegs

    @property
    def accumulation(self) -> NDArray[np.float64]:
        if self._accums is None:
            self._accums = compute_flow_accumulation(
                self.flowdir, self.dir_enc, valids=self.valid, indegs=self.indegree
            )
        return self._accums

    @property
    def strahler_order(self) -> NDArray[np.uint8]:
        if self._strahler_order is None:
            self._strahler_order = compute_flow_strahler_order(
                self.flowdir, self.dir_enc
            )
        return self._strahler_order

    def fill_depressions(self, max_fill_size: int | None = None) -> "DEMGrid":
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
        self, ocean_lvl: float = 0, min_size: int = 1, flood_below: bool = True
    ) -> "DEMGrid":
        """
        Marks sufficiently large boundary-connected ocean basins as
        invalid, which means they would not participate in
        calculation of flow directions, etc.

        Parameters
        ----------
        ocean_lvl : float, optional
            Elevation threshold defining ocean cells.
            Default elevation is `0`.
        min_size : int, optional
            Minimum cell count threshold for ocean basin
            invalidation.
            Ocean basins containing at least this number of cells
            are invalidated.
            Default size is `1`.
        flood_below : bool, optional
            Whether elevations strictly below `ocean_lvl` qualify
            as ocean cells.
            When false, only cells exactly equal to `ocean_lvl`
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
        prev_valid = self.valid.copy()
        self.valid = _invalidate_ocean_basins(
            self.dem,
            self.dir_enc,
            valids=self.valid,
            ocean_lvl=ocean_lvl,
            flood_below=flood_below,
            min_size=min_size,
        )

        newly_invalid = prev_valid & ~self.valid
        if self._ocean_mask is None:
            self._ocean_mask = newly_invalid
        else:
            self._ocean_mask |= newly_invalid
        self.ocean_threshold = ocean_lvl
        self._min_ocean_size = min_size
        self._ocean_flood_below = flood_below

        # Cached terrain products may depend directly or indirectly
        # on valid
        self._slope = None
        self._flat = None
        self._flat_gradient = None
        self._flowdir = None
        self._indegs = None
        self._accums = None
        self._strahler_order = None
        self._ws = None
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
    def dist2source(self) -> NDArray[np.float32]:
        if self._flowdist is None:
            self._flowdist = compute_dist2source(
                self.flowdir,
                self.dir_enc,
                x=self.x,
                y=self.y,
                valids=self.valid,
                indegs=self.indegree,
            )
        return self._flowdist

    @property
    def flow_distance(self) -> NDArray[np.float32]:
        return self.dist2source

    @property
    def watersheds(self) -> NDArray[np.int32]:
        if self._ws is not None:
            return self._ws

        self._ws = label_watersheds(self.flowdir, self.dir_enc, valids=self.valid)
        return self._ws

    @property
    def dist2sink(self) -> NDArray[np.float32]:
        if self._backdist is not None:
            return self._backdist

        self._backdist = compute_dist2sink(
            self.flowdir, self.dir_enc, x=self.x, y=self.y, valids=self.valid
        )
        return self._backdist

    @property
    def backdist(self) -> NDArray[np.float32]:
        return self.dist2sink

    @property
    def bmax(self) -> NDArray[np.float32]:
        if self._bmax is not None:
            return self._bmax

        self._bmax = compute_dist2conf_max(
            self.flowdir.astype(np.uint8, order="F"),
            self.dir_enc,
            self.valid.astype(np.bool_, order="F"),
            self.x.astype(np.float32, order="F"),
            self.y.astype(np.float32, order="F"),
        )
        return self._bmax

    @property
    def ridge_dist(self) -> NDArray[np.float32]:
        """
        'Distance' to the ridge, approximated by the distance to sink in the maximum confluence distance landscape.

        Returns
        -------
        dist : NDArray[np.float32]
            Distance to the ridge, approximated by the distance to sink in the maximum confluence distance landscape.
        """
        return self.dist2ridge

    @property
    def ridgedir(self) -> NDArray[np.uint8]:
        if self._ridgedir is not None:
            return self._ridgedir
        self._ridgedir = compute_ridgedir(
            self.flowdir, self.dir_enc, valids=self.valid, x=self.x, y=self.y
        )
        return self._ridgedir

    @property
    def dist2ridge(self) -> NDArray[np.float32]:
        """
        'Distance' to the ridge, approximated by the distance to sink in the maximum confluence distance landscape.

        Returns
        -------
        dist : NDArray[np.float32]
            Distance to the ridge, approximated by the distance to sink in the maximum confluence distance landscape.
        """

        if self._ridge_dist is not None:
            return self._ridge_dist

        self._ridge_dist = compute_dist2ridge(
            self.ridgedir,
            self.dir_enc,
            valids=self.valid.astype(np.bool_, order="F"),
            x=self.x.astype(np.float32, order="F"),
            y=self.y.astype(np.float32, order="F"),
            dir_is_ridge=True,
        )
        return self._ridge_dist

    @property
    def ridge_strahler_order(self) -> NDArray[np.uint8]:
        if self._ridge_strahler_order is not None:
            return self._ridge_strahler_order
        self._ridge_strahler_order = compute_ridge_strahler_order(
            self.ridgedir, self.dir_enc, valids=self.valid, dir_is_ridge=True
        )
        return self._ridge_strahler_order


def fill_pits(dem: NDArray[NpReal]) -> tuple[NDArray[NpReal], NDArray[np.bool_]]:
    """
    Notes
    -----
    This function is deprecated.
    """

    dem_filled = dem.copy()

    min_nabrs = np.min(get_neighbour_values(dem_filled)[0], axis=0)
    is_pit = dem_filled < min_nabrs
    dem_filled[is_pit] = min_nabrs[is_pit]

    return dem_filled, is_pit
