"""
Defines and validates raster flow-direction encoding schemes.

This module provides :class:`DirectionEncoding` and
:class:`D8DirectionEncoding`, which associate direction codes with
their corresponding row and column offsets.

Last modified: 2026-08-24, En-Chi Lee (williameclee@gmail.com)
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from formosa.utils import NpCanonIndex, NpFlowDir

names = ["self", "E", "SE", "S", "SW", "W", "NW", "N", "NE"]

Code = int | np.integer | NDArray[np.integer]
CodeTransf = Callable[[Code], Code]


@dataclass
class DirectionEncoding:
    codes: NDArray[NpFlowDir]
    offsets: NDArray[NpCanonIndex]

    @property
    def offset_dict(self) -> dict[int, tuple[int, int]]:
        """
        Dictionary where `offset_dict[code] = (di, dj)`.
        """
        return {
            int(code): (int(di), int(dj))
            for code, (di, dj) in zip(self.codes, self.offsets)
        }

    @property
    def offset_lookup(self) -> NDArray[NpCanonIndex]:
        """
        Array where `offset_lookup[code] = (di, dj)`.
        """

        offset_lookup = np.zeros((256, 2), dtype=NpCanonIndex)
        for code, offset in zip(self.codes, self.offsets):
            code = int(code)
            if not 0 <= code <= 255:
                raise ValueError(
                    f"Direction code must be between 0 and 255, got {code}."
                )
            offset_lookup[code] = offset
        return offset_lookup

    @property
    def code_lookup(self) -> NDArray[np.bool_]:
        """
        Array where `code_lookup[code] = True` if `code` is a valid
        code in the encoding, else `False`.
        """

        valid_lookup = np.zeros(256, dtype=bool)
        for code, offset in zip(self.codes, self.offsets):
            code = int(code)
            if not 0 <= code <= 255:
                raise ValueError(
                    f"Direction code must be between 0 and 255, got {code}."
                )
            valid_lookup[code] = True
        return valid_lookup

    @property
    def no_flow_code(self) -> int | None:
        """
        The code representing no flow (i.e. have the offset of `(0, 0)`).
        If such a code does not exist, returns `None`.
        """

        for code, (di, dj) in self.offset_dict.items():
            if di == 0 and dj == 0:
                return code
        return None

    @overload
    def code_to_offset(self, code: int) -> tuple[int, int]: ...

    @overload
    def code_to_offset(
        self, code: NDArray[NpFlowDir]
    ) -> tuple[NDArray[NpCanonIndex], NDArray[NpCanonIndex]]: ...

    def code_to_offset(self, code: Code) -> tuple[Code, Code]:
        """
        Gets offset (di, dj) for a given D8 code.
        """
        if isinstance(code, np.ndarray):
            return self._code_to_offset_ndarray(code)  # type: ignore
        elif isinstance(code, (int, np.integer)):
            return self._code_to_offset_scalar(code, self.offset_dict)  # type: ignore
        else:
            raise TypeError(f"Unsupported type for code: {type(code)}")

    def _code_to_offset_ndarray(
        self, code: NDArray[NpFlowDir]
    ) -> tuple[NDArray[NpCanonIndex], NDArray[NpCanonIndex]]:
        code = np.asarray(code)

        if np.issubdtype(code.dtype, np.integer):
            if code.dtype == np.uint8:
                safe_codes = code
                in_range = np.ones(code.shape, dtype=bool)
            else:
                in_range = (code >= 0) & (code <= 255)
                safe_codes = np.where(in_range, code, 0).astype(np.uint8, copy=False)

            known = in_range & self.code_lookup[safe_codes]
            didj = self.offset_lookup[safe_codes]
            didj[~known] = 0
            return didj[..., 0], didj[..., 1]

        # Floating fallback, preserving NaNs
        nan_mask = np.isnan(code)
        in_range = ~nan_mask & (code >= 0) & (code <= 255)
        safe_codes = np.where(in_range, code, 0).astype(NpFlowDir)

        known = in_range & self.code_lookup[safe_codes]
        didj = self.offset_lookup[safe_codes].astype(float)
        didj[~known] = 0
        didj[nan_mask] = np.nan

        return didj[..., 0], didj[..., 1]

    def _code_to_offset_scalar(
        self, code: int, offset_dict: dict[int, tuple[int, int]]
    ) -> tuple[int, int]:
        """Get offset (di, dj) for a given D8 code."""
        return offset_dict.get(code, (0, 0))


@dataclass
class D8DirectionEncoding(DirectionEncoding):
    window: int
    slices: int
    shape: str

    dirnames: list[str]

    def __init__(
        self,
        window: int = 3,
        slices: int = 8,
        shape: str = "circular",
        code_trans_func: CodeTransf | None = lambda x: 2 ** (x - 1),
        sort_by_dist: bool = True,
    ):
        self.window = window
        self.slices = slices
        self.shape = shape

        self.offsets, self.codes, self.dirnames = construct_d8_directions(
            window,
            slices,
            shape,
            code_transf_func=code_trans_func,
            sort_by_dist=sort_by_dist,
        )


def construct_d8_directions(
    window: int = 3,
    slices: int = 8,
    shape: str = "circular",
    dir_list: list[str] | None = names,
    code_transf_func: CodeTransf | None = lambda x: 2 ** (x - 1),
    sort_by_dist: bool = True,
) -> tuple[NDArray[NpCanonIndex], NDArray[NpFlowDir], list[str]]:
    assert window % 2 == 1, "Window size must be odd, got {window} instead"
    assert window >= 3, "Window size must be at least 3, got {window} instead"
    assert slices >= 2, "Number of slices must be at least 2, got {slices} instead"
    if dir_list is not None:
        assert len(dir_list) == slices + 1, (
            f"Number of names must be {slices + 1} (including self), "
            + f"got {len(dir_list)} instead"
        )
    if code_transf_func is None:
        code_transf_func = lambda x: x

    half_window: int = window // 2

    i: NDArray[np.integer] = np.arange(-half_window, half_window + 1, dtype=np.int32)
    j: NDArray[np.integer] = np.arange(-half_window, half_window + 1, dtype=np.int32)
    ii, jj = np.meshgrid(i, j, indexing="ij")

    az: NDArray[np.integer] = np.degrees(np.arctan2(ii, jj)) % 360
    az_agg: NDArray[np.integer] = np.mod(np.round(az * slices / 360), slices) + 1
    az_agg[half_window, half_window] = 0  # centre pixel

    dists: NDArray[np.integer] = ii**2 + jj**2

    if shape == "circular":
        mask = dists > (window / 2) ** 2
        az_agg[mask] = -1

    offsets = np.array([ii.flatten(), jj.flatten()], dtype=NpCanonIndex).T
    codes = np.zeros(az_agg.shape, dtype=NpFlowDir)
    codes[az_agg > 0] = code_transf_func(az_agg[az_agg > 0])
    codes = codes.flatten()
    offsets = offsets[az_agg.flatten() >= 0]
    codes = codes[az_agg.flatten() >= 0]

    # Check for duplicate codes
    uniq_codes, cnts = np.unique(codes, return_counts=True)
    dup_codes = uniq_codes[cnts > 1]
    if len(dup_codes) > 0:
        raise ValueError(f"Duplicate codes found: {dup_codes}")

    if sort_by_dist:
        dists = dists.flatten()[az_agg.flatten() >= 0]
        offsets = offsets[np.argsort(dists)]
        codes = codes[np.argsort(dists)]

    if dir_list is not None:
        name_dict = {
            code: name
            for code, name in zip(
                [0]
                + list(
                    map(
                        code_transf_func,
                        [i for i in range(1, slices + 1)],
                    )
                ),
                dir_list,
            )
        }
        dirs = [name_dict[code] for code in codes]
    else:
        dirs = []

    return offsets, codes, dirs


def validate_direction_offsets(
    ofsts: ArrayLike,
) -> NDArray[np.int32]:
    """
    Validates and formats row-column connectivity offsets.
    """
    ofsts = np.asarray(ofsts)
    if ofsts.ndim != 2 or ofsts.shape[1] != 2:
        raise ValueError(
            "Direction offsets must have shape (n, 2), "
            + f"but got shape {ofsts.shape}."
        )
    if ofsts.shape[0] == 0:
        raise ValueError("Direction offsets must contain at least one offset.")
    if not np.issubdtype(ofsts.dtype, np.integer):
        raise TypeError(
            "Direction offsets must have an integer dtype, " + f"but got {ofsts.dtype}."
        )

    int32_limits = np.iinfo(np.int32)
    if np.any((ofsts < int32_limits.min) | (ofsts > int32_limits.max)):
        raise ValueError("Direction offsets must be representable as int32 values.")
    return np.asfortranarray(ofsts, dtype=np.int32)
