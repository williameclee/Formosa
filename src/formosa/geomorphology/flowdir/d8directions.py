# Last modified
#      - Moved from `geomorphology.flowdir` to `geomorphology.flowdir.flowdir`
#   2026-08-03, En-Chi Lee (williameclee@gmail.com)
#     - Added property `no_flow_code`
#     - Got rid of built-in list iteration to accelerate `_code2offset_ndarray`

import numpy as np

from typing import TypeVar, Tuple, Callable, Optional
import numpy.typing as npt

names = ["self", "E", "SE", "S", "SW", "W", "NW", "N", "NE"]

T = TypeVar("T", int, np.integer, npt.NDArray[np.integer])


class D8Directions:
    def __init__(
        self,
        window: int = 3,
        slices: int = 8,
        shape: str = "circular",
        transform_codes: Callable | None = lambda x: 2 ** (x - 1),
        sort_by_distance: bool = True,
    ):
        self.window = window
        self.slices = slices
        self.shape = shape

        self.offsets, self.codes, self.dirnames = construct_d8_directions(
            window=window,
            slices=slices,
            shape=shape,
            code_transform_func=transform_codes,
            sort_by_distance=sort_by_distance,
        )

        self.offset_dict = {
            int(code): (int(di), int(dj))
            for code, (di, dj) in zip(self.codes, self.offsets)
        }

        self.offset_lookup = np.zeros((256, 2), dtype=np.int32)
        self.valid_code_lookup = np.zeros(256, dtype=bool)

        for code, offset in zip(self.codes, self.offsets):
            code = int(code)
            if not 0 <= code <= 255:
                raise ValueError(
                    f"Direction code must be between 0 and 255, got {code}."
                )
            self.offset_lookup[code] = offset
            self.valid_code_lookup[code] = True

    @property
    def no_flow_code(self) -> Optional[int]:
        """
        The code representing no flow (i.e. have the offset of `(0, 0)`).
        If such a code does not exist, returns `None`.
        """

        for code, (di, dj) in self.offset_dict.items():
            if di == 0 and dj == 0:
                return code
        return None

    def code2d8offset(self, code: T) -> tuple[T, T]:
        """Get offset (di, dj) for a given D8 code."""
        if isinstance(code, np.ndarray):
            return self._code2offset_ndarray(code)  # type: ignore
        elif isinstance(code, (int, np.integer)):
            return self._code2offset_scalar(code, self.offset_dict)  # type: ignore
        else:
            raise TypeError(f"Unsupported type for code: {type(code)}")

    def _code2offset_ndarray(
        self, code: npt.NDArray[np.integer]
    ) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]]:
        code = np.asarray(code)

        if np.issubdtype(code.dtype, np.integer):
            if code.dtype == np.uint8:
                safe_codes = code
                in_range = np.ones(code.shape, dtype=bool)
            else:
                in_range = (code >= 0) & (code <= 255)
                safe_codes = np.where(in_range, code, 0).astype(np.uint8, copy=False)

            known = in_range & self.valid_code_lookup[safe_codes]
            didj = self.offset_lookup[safe_codes]
            didj[~known] = 0
            return didj[..., 0], didj[..., 1]

        # Floating fallback, preserving NaNs
        nan_mask = np.isnan(code)
        in_range = ~nan_mask & (code >= 0) & (code <= 255)
        safe_codes = np.where(in_range, code, 0).astype(np.uint8)

        known = in_range & self.valid_code_lookup[safe_codes]
        didj = self.offset_lookup[safe_codes].astype(float)
        didj[~known] = 0
        didj[nan_mask] = np.nan

        return didj[..., 0], didj[..., 1]

    def _code2offset_scalar(
        self, code: int, offset_dict: dict[int, tuple[int, int]]
    ) -> tuple[int, int]:
        """Get offset (di, dj) for a given D8 code."""
        return offset_dict.get(code, (0, 0))


def construct_d8_directions(
    window: int = 3,
    slices: int = 8,
    shape: str = "circular",
    dir_list: list[str] | None = names,
    code_transform_func: Callable | None = lambda x: 2 ** (x - 1),
    sort_by_distance: bool = True,
) -> Tuple[npt.NDArray[np.integer], npt.NDArray[np.integer], list[str]]:
    assert window % 2 == 1, "Window size must be odd, got {window} instead"
    assert window >= 3, "Window size must be at least 3, got {window} instead"
    assert slices >= 2, "Number of slices must be at least 2, got {slices} instead"
    if dir_list is not None:
        assert (
            len(dir_list) == slices + 1
        ), f"Number of names must be {slices + 1} (including self), got {len(dir_list)} instead"
    if code_transform_func is None:
        code_transform_func = lambda x: x

    half_window: int = window // 2

    i: npt.NDArray[np.integer] = np.arange(
        -half_window, half_window + 1, dtype=np.int32
    )
    j: npt.NDArray[np.integer] = np.arange(
        -half_window, half_window + 1, dtype=np.int32
    )
    ii, jj = np.meshgrid(i, j, indexing="ij")

    az: npt.NDArray[np.integer] = np.degrees(np.arctan2(ii, jj)) % 360
    az_agg: npt.NDArray[np.integer] = np.mod(np.round(az * slices / 360), slices) + 1
    az_agg[half_window, half_window] = 0  # centre pixel

    dists: npt.NDArray[np.integer] = ii**2 + jj**2

    if shape == "circular":
        mask = dists > (window / 2) ** 2
        az_agg[mask] = -1

    offsets: npt.NDArray[np.integer] = np.array([ii.flatten(), jj.flatten()]).T
    codes: npt.NDArray[np.integer] = np.zeros(az_agg.shape, dtype=np.int16)
    codes[az_agg > 0] = code_transform_func(az_agg[az_agg > 0])
    codes = codes.flatten()
    offsets = offsets[az_agg.flatten() >= 0]
    codes = codes[az_agg.flatten() >= 0]

    # Check for duplicate codes
    unique_codes, counts = np.unique(codes, return_counts=True)
    duplicate_codes = unique_codes[counts > 1]
    if len(duplicate_codes) > 0:
        raise ValueError(f"Duplicate codes found: {duplicate_codes}")

    if sort_by_distance:
        dists = dists.flatten()[az_agg.flatten() >= 0]
        offsets = offsets[np.argsort(dists)]
        codes = codes[np.argsort(dists)]

    if dir_list is not None:
        name_dict = {
            code: name
            for code, name in zip(
                [0] + list(map(code_transform_func, [i for i in range(1, slices + 1)])),
                dir_list,
            )
        }
        dirs = [name_dict[code] for code in codes]
    else:
        dirs = []

    codes = codes.astype(np.int32, order="F")
    return offsets, codes, dirs
