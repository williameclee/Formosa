import numpy as np

from typing import TypeVar, Tuple, Callable
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
        # assert (
        #     codes.shape[0] == offsets.shape[0]
        # ), f"Length of codes and offsets must ber equal, got {codes.shape[0]} and {offsets.shape[0]} instead"
        # self.codes = codes
        # self.offsets = offsets
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

    def code2d8offset(self, code: T) -> tuple[T, T]:
        """Get offset (di, dj) for a given D8 code."""
        if isinstance(code, np.ndarray):
            return self._code2offset_ndarray(code, self.offset_dict)  # type: ignore
        elif isinstance(code, (int, np.integer)):
            return self._code2offset_scalar(code, self.offset_dict)  # type: ignore
        else:
            raise TypeError(f"Unsupported type for code: {type(code)}")

    def _code2offset_ndarray(
        self,
        code: npt.NDArray[np.integer],
        offset_dict: dict[int, tuple[int, int]],
    ) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]]:
        """Get offset (di, dj) for a given D8 code."""
        didj = np.array(
            [
                offset_dict.get(int(c), (0, 0)) if not np.isnan(c) else (-999, -999)
                for c in code.flatten()
            ],
            dtype=self.offsets.dtype,
        ).reshape(code.shape + (2,))
        # replace (-999, -999) back to np.nan
        didj = np.where(didj == -999, np.nan, didj)
        di = didj[..., 0]
        dj = didj[..., 1]
        return di, dj

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


if __name__ == "__main__":
    flowdir = np.array([[0, 1], [4, 16]], dtype=np.int32)  # 2x2 array with D8 codes
    di, dj = D8Directions(window=5).code2d8offset(flowdir)
    print(f"di:\n{di}\ndj:\n{dj}")
    # offsets, codes, names = construct_d8_directions(
    #     window=3, slices=8, shape="circular"
    # )
    # print("Offsets:\n", offsets)
    # print("Codes:\n", codes)
    # print("Names:\n", names)

# plt.imshow(codes)
# plt.colorbar()
# plt.show()
