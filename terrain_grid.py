import numpy as np
import rasterio
import scipy.ndimage as ndi
from skimage import morphology

import d8directions
import flowdir
import demio

import numpy.typing as npt


class DEMGrid:
    def __init__(
        self,
        dem: npt.NDArray[np.number] | str,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        transform: rasterio.Affine | None = None,
        gaussian_filter: float | None = None,
        stride: int | None = None,
        detect_ocean: bool | float | int = False,
        directions: d8directions.D8Directions = d8directions.D8Directions(),
    ):
        if isinstance(dem, str):
            dem, x, y, transform = demio.read_dem(dem)
            self._original_dem = dem
            self.dem = dem
            self.x = x
            self.y = y
            self.transform = transform
        else:
            self._original_dem = dem
            self.dem = dem
            self.x = x
            self.y = y
            self.transform = (
                transform if transform is not None else rasterio.Affine.identity()
            )

        if stride is not None:
            assert (
                stride > 0
            ), f"STRIDE must be a positive integer, got {stride} instead"
            self.num_slice = stride
            self.dem = self.dem[::stride, ::stride]
            if self.x is not None:
                self.x = self.x[::stride]
            if self.y is not None:
                self.y = self.y[::stride]
        else:
            self.num_slice = 1

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
        self._flat: None | npt.NDArray[np.bool] = None
        self._flat_gradient: None | npt.NDArray[np.integer] = None
        self._flowdir: None | npt.NDArray[np.integer] = None
        self._indegree: None | npt.NDArray[np.integer] = None
        self._accumulation: None | npt.NDArray[np.integer] = None
        self._strahler_order: None | npt.NDArray[np.integer] = None
        self._graphx = None
        self._graphy = None

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
        if self._flowdir is None or self._flat is None:
            self._flowdir, self._flat, self._flat_gradient = flowdir.compute_flowdir(
                self.dem,
                directions=self.directions,
                resolve_flat=True,
            )
        return self._flowdir

    @property
    def flowdir_graph_xy(
        self,
    ) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]]:
        if self._graphx is None or self._graphy is None:
            self._graphy, self._graphx = flowdir.compute_flowdir_graph(
                self.flowdir, directions=self.directions
            )
        return self._graphx, self._graphy

    @property
    def indegree(self) -> npt.NDArray[np.integer]:
        if self._indegree is None:
            self._indegree = flowdir.compute_indegree(
                self.flowdir, directions=self.directions
            )
        return self._indegree

    @property
    def accumulation(self) -> np.ndarray:
        if self._accumulation is None:
            self._accumulation = flowdir.compute_accumulation(
                self.flowdir,
                valids=self.valid,
                indegrees=self.indegree,
                directions=self.directions,
            )
        return self._accumulation

    @property
    def strahler_order(self) -> npt.NDArray[np.integer]:
        if self._strahler_order is None:
            self._strahler_order = flowdir.compute_strahler_order(
                self.flowdir,
                directions=self.directions,
            )
        return self._strahler_order

    def fill_depressions(self):
        self.dem = fill_depressions(fill_pits(self.dem)[0], valid=self.valid)
        return self


def detect_ocean_mask(dem, ocean_threshold: int | float = 0):
    neighbours, _, _ = flowdir.get_neighbour_values(dem, include_self=False)
    ocean_mask = (dem <= ocean_threshold) & np.any(
        neighbours <= ocean_threshold, axis=0
    )
    return ocean_mask


def fill_pits(
    dem: npt.NDArray[np.number],
) -> tuple[npt.NDArray[np.number], npt.NDArray[np.bool]]:
    dem_filled = dem.copy()

    min_neighbours = np.min(flowdir.get_neighbour_values(dem_filled)[0], axis=0)
    is_pit = dem_filled < min_neighbours
    dem_filled[is_pit] = min_neighbours[is_pit]

    return dem_filled, is_pit


def fill_depressions(
    dem: npt.NDArray[np.number],
    valid: npt.NDArray[np.bool] | None = None,
) -> npt.NDArray[np.number]:
    dem_seed = dem.copy()
    if valid is not None:
        dem[~valid] = np.nanmin(dem[valid])
        seed_value = np.nanmax(dem[valid]) + 1
    else:
        seed_value = np.nanmax(dem) + 1

    dem_mask = np.full(dem.shape, True, dtype=np.bool)
    dem_mask[0, :] = False
    dem_mask[-1, :] = False
    dem_mask[:, 0] = False
    dem_mask[:, -1] = False
    dem_seed[dem_mask] = seed_value
    return morphology.reconstruction(dem_seed, dem, method="erosion").astype(dem.dtype)
