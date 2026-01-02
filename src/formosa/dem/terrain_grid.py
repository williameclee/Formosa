from pathlib import Path
import warnings
import numpy as np
import rasterio
import rasterio.transform as rt
import scipy.ndimage as ndi
from skimage import morphology

from formosa.dem import read_dem
from formosa.geomorphology.d8directions import D8Directions
from formosa.geomorphology.flowdir import (
    compute_flowdir,
    compute_flowdir_graph,
    compute_indegree,
    compute_accumulation,
    compute_strahler_order,
    compute_flow_distance,
    get_neighbour_values,
)

import numpy.typing as npt


class DEMGrid:
    _original_dem: npt.NDArray[np.number]
    dem: npt.NDArray[np.number]
    x: npt.NDArray[np.floating | np.integer]
    y: npt.NDArray[np.floating | np.integer]
    transform: rasterio.Affine
    i: npt.NDArray[np.uint32]
    j: npt.NDArray[np.uint32]
    valid: npt.NDArray[np.bool]

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
    ):
        if isinstance(dem, (str, Path)):
            # Read from a tiff file
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
        self._sea_mask = None
        self.sea_threshold = None
        if detect_ocean is not False:
            self.sea_threshold = (
                detect_ocean if isinstance(detect_ocean, (int, float)) else 0
            )
            self._sea_mask = detect_ocean_mask(self.dem, self.sea_threshold)
            self.valid = self.valid & ~self._sea_mask

        self.gaussian_filter = gaussian_filter
        self.directions = directions
        if gaussian_filter is not None:
            filtered_dem = ndi.gaussian_filter(self.dem, sigma=gaussian_filter)
            self.dem = np.where(self.valid, filtered_dem, self.dem)

        self.quality = np.zeros(self.dem.shape, dtype=np.int16)
        self._slope: None | npt.NDArray[np.integer | np.floating] = None
        self._flat: None | npt.NDArray[np.bool] = None
        self._flat_gradient: None | npt.NDArray[np.integer] = None
        self._flowdir: None | npt.NDArray[np.integer] = None
        self._indegree: None | npt.NDArray[np.integer] = None
        self._accumulation: None | npt.NDArray[np.integer] = None
        self._strahler_order: None | npt.NDArray[np.integer] = None
        self._watershed: None | npt.NDArray[np.int32] = None
        self._graphx = None
        self._graphy = None
        self._backdist: None | npt.NDArray[np.float32] = None
        self._flowdist: None | npt.NDArray[np.integer] = None

    @property
    def slope(self) -> npt.NDArray[np.floating | np.integer]:
        if self._slope is not None:
            return self._slope

        from formosa.geomorphology.terrain import compute_slope

        self._slope = compute_slope(self.dem, x=self.x, y=self.y)
        self._slope[~self.valid] = np.nan
        return self._slope

    @property
    def sea_mask(self) -> npt.NDArray[np.bool]:
        if self._sea_mask is None or self.sea_threshold is None:
            if self.sea_threshold is None:
                self.sea_threshold = 0
            sea_mask = detect_ocean_mask(self.dem, self.sea_threshold)
            self._sea_mask = sea_mask & self.valid
            self.valid = self.valid & ~self._sea_mask

        return self._sea_mask

    @property
    def flowdir(self) -> npt.NDArray[np.integer]:
        if self._flowdir is None:
            self._flowdir, self._flat, self._flat_gradient = compute_flowdir(
                self.dem,
                directions=self.directions,
                resolve_flat=True,
            )
        return self._flowdir

    def flowdir_graph_xy(
        self,
        valid: npt.NDArray[np.bool] | None = None,
    ) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]]:
        graphy, graphx = compute_flowdir_graph(
            self.flowdir,
            directions=self.directions,
            valid=valid if valid is not None else self.valid,
            x=self.x,
            y=self.y,
        )
        return graphx, graphy

    @property
    def indegree(self) -> npt.NDArray[np.integer]:
        if self._indegree is None:
            self._indegree = compute_indegree(self.flowdir, directions=self.directions)
        return self._indegree

    @property
    def accumulation(self) -> np.ndarray:
        if self._accumulation is None:
            self._accumulation = compute_accumulation(
                self.flowdir,
                valids=self.valid,
                indegrees=self.indegree,
                directions=self.directions,
            )
        return self._accumulation

    @property
    def strahler_order(self) -> npt.NDArray[np.integer]:
        if self._strahler_order is None:
            self._strahler_order = compute_strahler_order(
                self.flowdir,
                directions=self.directions,
            )
        return self._strahler_order

    def fill_depressions(self, method: str = "erosion") -> "DEMGrid":
        self.dem = fill_depressions(
            fill_pits(self.dem)[0], valid=self.valid, method=method
        )
        return self

    @property
    def flow_distance(self) -> npt.NDArray[np.integer]:
        if self._flowdist is None:
            self._flowdist = compute_flow_distance(
                self.flowdir,
                directions=self.directions,
            )
        return self._flowdist

    @property
    def watersheds(self) -> npt.NDArray[np.int32]:
        if self._watershed is not None:
            return self._watershed

        from formosa.geomorphology.flowdir import label_watersheds

        self._watershed = label_watersheds(
            self.flowdir,
            directions=self.directions,
            valids=self.valid,
        )
        return self._watershed

    @property
    def backdist(self) -> npt.NDArray[np.float32]:
        if self._backdist is not None:
            return self._backdist

        from formosa.geomorphology.flowdir import compute_back_distance

        self._backdist = compute_back_distance(
            self.flowdir,
            directions=self.directions,
            valids=self.valid,
        )
        return self._backdist


def detect_ocean_mask(dem, ocean_threshold: int | float = 0):
    neighbours, _, _ = get_neighbour_values(dem, include_self=False)
    ocean_mask = (dem <= ocean_threshold) & np.any(
        neighbours <= ocean_threshold, axis=0
    )
    return ocean_mask


def fill_pits(
    dem: npt.NDArray[np.number],
) -> tuple[npt.NDArray[np.number], npt.NDArray[np.bool]]:
    dem_filled = dem.copy()

    min_neighbours = np.min(get_neighbour_values(dem_filled)[0], axis=0)
    is_pit = dem_filled < min_neighbours
    dem_filled[is_pit] = min_neighbours[is_pit]

    return dem_filled, is_pit


def fill_depressions(
    dem: npt.NDArray[np.number],
    valid: npt.NDArray[np.bool] | None = None,
    method: str = "erosion",
) -> npt.NDArray[np.number]:
    assert method in [
        "erosion",
        "dilation",
    ], f"METHOD must be either 'erosion' or 'dilation', got {method} instead"

    dem_seed = dem.copy()
    if valid is not None:
        if method == "erosion":
            dem[~valid] = np.nanmin(dem[valid])
            seed_value = np.nanmax(dem[valid]) + 1
        else:
            dem[~valid] = np.nanmax(dem[valid])
            seed_value = np.nanmin(dem[valid]) - 1
    else:
        if method == "erosion":
            seed_value = np.nanmax(dem) + 1
        else:
            seed_value = np.nanmin(dem) - 1

    dem_mask = np.full(dem.shape, True, dtype=np.bool)
    dem_mask[0, :] = False
    dem_mask[-1, :] = False
    dem_mask[:, 0] = False
    dem_mask[:, -1] = False
    dem_seed[dem_mask] = seed_value
    return morphology.reconstruction(dem_seed, dem, method=method).astype(dem.dtype)
