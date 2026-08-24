"""
Labels watershed rasters using the Python backend.

This module implements internal routines called by the public-facing
drainage API and is not intended to be used directly.

Last modified: 2026-08-24, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np
from numpy.typing import NDArray

from formosa.geomorphology.drainage.directions import DirectionEncoding
from formosa.utils import NpFlowDir


def label_watersheds(
    dirs: NDArray[NpFlowDir], dir_enc: DirectionEncoding, valids: NDArray[np.bool_]
) -> NDArray[np.int32]:
    I, J = dirs.shape
    ii, jj = np.meshgrid(
        np.arange(I, dtype=np.int32), np.arange(J, dtype=np.int32), indexing="ij"
    )
    codes: list[int] = dir_enc.codes.tolist()
    offsets: list[tuple[int, int]] = [
        (int(di), int(dj)) for di, dj in dir_enc.offsets.astype(np.int32, copy=False)
    ]

    seeds: list[tuple[int, int]] = list(
        zip(ii[valids & (dirs == 0)], jj[valids & (dirs == 0)])
    )

    ws = -np.ones(dirs.shape, dtype=np.int32)

    for label, seed in enumerate(seeds):
        to_fill: list[tuple[int, int]] = [seed]

        while to_fill:
            ci, cj = to_fill.pop(0)
            ws[ci, cj] = label
            for code, (di, dj) in zip(codes, offsets):
                ni, nj = ci - di, cj - dj
                if (ni < 0 or ni >= I) or (nj < 0 or nj >= J):
                    continue
                elif not valids[ni, nj]:
                    continue
                elif ws[ni, nj] != -1:
                    continue

                if dirs[ni, nj] == code:
                    to_fill.append((ni, nj))
    ws = ws + 1  # make background 0 and watersheds start from 1
    return ws
