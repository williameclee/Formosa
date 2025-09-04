import numpy as np
from collections import deque

# Custom functions
from d8directions import D8Directions

# Misc
import matplotlib.pyplot as plt

# type check
from typing import Union
import numpy.typing as npt


def _compute_flowdir_simple(
    dem: npt.NDArray[np.number],
    directions: D8Directions = D8Directions(),
) -> npt.NDArray[np.integer]:
    is_low_flat = find_flat(dem, directions=directions)
    flat_neighbours, _, _ = get_neighbour_values(
        is_low_flat, directions=D8Directions(window=3), include_self=False
    )
    is_1px_flat = is_low_flat & ~np.any(flat_neighbours, axis=0)

    neighbours, codes, offsets = get_neighbour_values(
        dem, directions=directions, include_self=True, pad_value=np.max(dem) + 1
    )
    # neighbours[0][is_1px_flat] = np.max(neighbours) + 1
    flowdir = np.nanargmin(neighbours, axis=0)

    is_ambiguous = find_ambiguous(dem, directions=directions)
    is_ambiguous = is_ambiguous & ~is_1px_flat

    flowdir = codes[flowdir].astype(np.integer)
    return flowdir


def find_flat_edges(
    dem: npt.NDArray[np.number],
    flowdir: npt.NDArray[np.integer],
    directions=D8Directions(),
) -> tuple[npt.NDArray[np.bool], npt.NDArray[np.bool]]:
    """
    From R. Barnes et al., 2014, Algorithm 3 (p. 133)
    """
    neighbours, _, _ = get_neighbour_values(
        dem,
        directions=directions,
        include_self=False,
        pad_value=np.min(dem) - 1,  # since is_high_edge
    )
    neighbour_flowdirs, _, _ = get_neighbour_values(
        flowdir, directions=directions, include_self=False, pad_value=-1
    )

    is_high_edge: npt.NDArray[np.bool] = (flowdir == 0) & np.any(
        dem < neighbours, axis=0
    )
    is_low_edge: npt.NDArray[np.bool] = (flowdir != 0) & np.any(
        (neighbour_flowdirs == 0) & (dem == neighbours), axis=0
    )

    return is_low_edge, is_high_edge


def label_flats(
    dem: npt.NDArray[np.number],
    low_edges: npt.NDArray[np.bool] | deque[tuple[int, int]],
    directions: D8Directions = D8Directions(),
) -> tuple[npt.NDArray[np.uint32], int, int]:
    """
    From R. Barnes et al., 2014, Algorithm 4 (p. 133)
    """
    ## Input validation and initialisation
    if isinstance(low_edges, np.ndarray):
        assert (
            dem.shape == low_edges.shape
        ), f"DEM and LOW_EDGES must have the same shape, got {dem.shape} and {low_edges.shape} instead"

        ii, jj = np.indices(dem.shape, dtype=np.uint16)
        low_edges = deque(zip(ii[low_edges], jj[low_edges]))

    ## Initialisation
    I, J = dem.shape
    nanfill = 0
    labels: npt.NDArray[np.uint32] = np.full(dem.shape, nanfill)
    label = 1

    ## Main
    while low_edges:
        si, sj = low_edges.popleft()

        if labels[si, sj] != nanfill:
            continue

        height = dem[si, sj]
        to_fill: deque[tuple[int, int]] = deque()
        to_fill.append((si, sj))

        while to_fill:
            i, j = to_fill.popleft()

            if i < 0 or i >= I or j < 0 or j >= J:
                continue
            if labels[i, j] != nanfill:
                continue
            if dem[i, j] != height:
                continue
            labels[i, j] = label

            for di, dj in directions.offsets:
                to_fill.append((i + di, j + dj))
        label += 1

    num_labels = label - 1
    return labels, num_labels, nanfill


def get_neighbour_values(
    array: np.ndarray,
    directions: D8Directions = D8Directions(),
    pad_value: np.number | float | int = np.nan,
    include_self: bool = False,
    self_at_last: bool = False,
) -> tuple[np.ndarray, npt.NDArray[np.integer], npt.NDArray[np.integer]]:
    ## Input validation and initialisation
    if np.issubdtype(array.dtype, np.integer) and pad_value is np.nan:
        Warning("Integer array does not support NaN padding, using max int instead")
        pad_value = np.iinfo(array.dtype).max

    ## Main
    # get padding width from offset
    pad_width = np.max(abs(directions.offsets))
    array_padded = np.pad(
        array,
        pad_width=pad_width,
        mode="constant",
        constant_values=pad_value,
    )
    neighbours = np.zeros((len(directions.codes), *array.shape), dtype=array.dtype)
    offsets = np.zeros((len(directions.codes), 2), dtype=np.integer)
    for i_offset, [di, dj] in enumerate(directions.offsets.astype(np.int16)):
        offsets[i_offset, :] = [di, dj]
        neighbours[i_offset, :, :] = array_padded[
            pad_width + di : pad_width + di + array.shape[0],
            pad_width + dj : pad_width + dj + array.shape[1],
        ]

    codes = directions.codes
    if not include_self:
        # exclude self (first offset)
        self_id = np.where(np.all(directions.offsets == [0, 0], axis=1))[0][0]
        neighbours = np.delete(neighbours, self_id, axis=0)
        codes = np.delete(codes, self_id, axis=0)
        offsets = np.delete(offsets, self_id, axis=0)
    elif self_at_last:
        neighbours = np.roll(neighbours, -1, axis=0)
        codes = np.roll(codes, -1, axis=0)
        offsets = np.roll(offsets, -1, axis=0)
    return neighbours, codes, offsets


def compute_downstream_indices(
    flowdirs: npt.NDArray[np.integer],
    directions: D8Directions = D8Directions(),
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer], npt.NDArray[np.integer]]:
    I, J = flowdirs.shape
    ii, jj = np.meshgrid(
        np.arange(I, dtype=np.integer), np.arange(J, dtype=np.integer), indexing="ij"
    )
    di, dj = directions.code2d8offset(flowdirs)
    dsi = (ii.astype(np.int16) + (di).astype(np.int16)).astype(np.integer)
    dsj = (jj.astype(np.int16) + (dj).astype(np.int16)).astype(np.integer)
    dsij = dsj.astype(np.integer) * I + dsi.astype(np.integer)
    return dsi, dsj, dsij


def find_ambiguous(
    dem: npt.NDArray[np.number],
    directions: D8Directions = D8Directions(),
) -> npt.NDArray[np.bool]:
    neighbours, _, _ = get_neighbour_values(dem, directions=directions)
    min_neighbours = np.min(neighbours, axis=0)
    is_ambiguous = np.sum(neighbours == min_neighbours, axis=0) > 1
    is_ambiguous = is_ambiguous & ~(find_flat(dem))
    return is_ambiguous


def find_flat(
    dem: npt.NDArray[np.number],
    valid: npt.NDArray[np.bool] | None = None,
    only_min: bool = True,
    directions: D8Directions = D8Directions(window=3),
) -> npt.NDArray[np.bool]:
    if valid is not None and np.any(~valid):
        dem[~valid] = np.max(dem[~valid]) + 1

    neighbours, _, _ = get_neighbour_values(
        dem, directions=directions, pad_value=np.nan, include_self=False
    )
    if only_min:
        is_flat = dem == np.nanmin(neighbours, axis=0)
    else:
        is_flat = np.any(dem == neighbours, axis=0)

    if valid is not None:
        is_flat = is_flat & valid
    return is_flat


def compute_away_from_high(
    flowdirs: npt.NDArray[np.number],
    labels: npt.NDArray[np.number],
    high_edges: npt.NDArray[np.bool] | deque[tuple[int, int]],
    directions: D8Directions = D8Directions(),
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]]:
    """
    From R. Barnes et al., 2014, Algorithm 5 (p. 133–134)
    """
    ## Input validation and initialisation
    assert (
        flowdirs.shape == labels.shape
    ), f"FLOWDIRS and LABELS must have the same shape, got {flowdirs.shape} and {labels.shape} instead"

    if isinstance(high_edges, np.ndarray):
        assert (
            flowdirs.shape == high_edges.shape
        ), f"FLOWDIRS and HIGH_EDGES must have the same shape, got {flowdirs.shape} and {high_edges.shape} instead"

        ii, jj = np.indices(flowdirs.shape, dtype=np.integer)
        high_edges = deque(zip(ii[high_edges], jj[high_edges]))

    flat_mask = np.zeros(flowdirs.shape, dtype=np.integer)
    flat_height = np.zeros((np.nanmax(labels) + 1,), dtype=np.integer)

    marker: tuple[int, int] = (-1, -1)
    high_edges.append(marker)

    loops: int = 1

    while len(high_edges) > 1:
        ci, cj = high_edges.popleft()

        if (ci, cj) == marker:
            loops += 1
            high_edges.append(marker)
            continue

        if flat_mask[ci, cj] > 0:
            continue

        flat_mask[ci, cj] = loops
        flat_height[labels[ci, cj]] = loops

        for di, dj in directions.offsets:
            ni, nj = ci + di, cj + dj
            if ni < 0 or ni >= flowdirs.shape[0] or nj < 0 or nj >= flowdirs.shape[1]:
                continue
            if labels[ni, nj] != labels[ci, cj]:
                continue
            if flowdirs[ni, nj] != 0:
                continue

            high_edges.append((ni, nj))
    return flat_mask, flat_height


def compute_towards_low(
    flowdirs: npt.NDArray[np.number],
    labels: npt.NDArray[np.number],
    flat_mask: npt.NDArray[np.integer],
    low_edges: npt.NDArray[np.bool] | deque[tuple[int, int]],
    flat_height: npt.NDArray[np.number],
    directions: D8Directions = D8Directions(),
    step_size: int = 2,
) -> npt.NDArray[np.integer]:
    """
    From R. Barnes et al., 2014, Algorithm 6 (p. 134)
    """
    ## Input validation and initialisation
    assert (
        flowdirs.shape == labels.shape
    ), f"FLOWDIRS and LABELS must have the same shape, got {flowdirs.shape} and {labels.shape} instead"
    assert (
        flowdirs.shape == flat_mask.shape
    ), f"FLOWDIRS and FLAT_MASK must have the same shape, got {flowdirs.shape} and {flat_mask.shape} instead"

    assert flat_height.shape == (
        np.nanmax(labels) + 1,
    ), f"FLATHEIGHT must have shape ({np.nanmax(labels) + 1},), got {flat_height.shape} instead"

    assert (
        step_size > 0
    ), f"STEPSIZE must be a positive integer, got {step_size} instead"

    if isinstance(low_edges, np.ndarray):
        assert (
            flowdirs.shape == low_edges.shape
        ), f"FLOWDIRS and HIGH_EDGES must have the same shape, got {flowdirs.shape} and {low_edges.shape} instead"

        ii, jj = np.indices(flowdirs.shape, dtype=np.integer)
        low_edges = deque(zip(ii[low_edges], jj[low_edges]))

    flat_mask = -flat_mask
    flat_height = np.zeros((np.nanmax(labels) + 1,), dtype=labels.dtype)

    marker: tuple[int, int] = (-1, -1)
    low_edges.append(marker)

    ## Main
    loops: int = 1

    while len(low_edges) > 1:
        ci, cj = low_edges.popleft()

        if (ci, cj) == marker:
            loops += 1
            low_edges.append(marker)
            continue

        if flat_mask[ci, cj] > 0:
            continue

        elif flat_mask[ci, cj] < 0:
            flat_mask[ci, cj] += flat_height[labels[ci, cj]] + loops * step_size
        else:
            flat_mask[ci, cj] = loops * step_size

        for di, dj in directions.offsets:
            ni, nj = ci + di, cj + dj
            if ni < 0 or ni >= flowdirs.shape[0] or nj < 0 or nj >= flowdirs.shape[1]:
                continue
            if labels[ni, nj] != labels[ci, cj]:
                continue
            if flowdirs[ni, nj] != 0:
                continue

            low_edges.append((ni, nj))
    return flat_mask


def compute_masked_flowdir(
    flat_mask, labels, directions: D8Directions = D8Directions()
):
    """
    From R. Barnes et al., 2014, Algorithm 7 (p. 134)
    """

    flowdir = np.zeros(flat_mask.shape, dtype=np.integer)

    for ci in range(flat_mask.shape[0]):
        for cj in range(flat_mask.shape[1]):
            if labels[ci, cj] == 0:
                continue

            dir = 0
            fmin = flat_mask[ci, cj]

            for di, dj, code in zip(
                directions.offsets[:, 0], directions.offsets[:, 1], directions.codes
            ):
                ni, nj = ci + di, cj + dj
                if (
                    ni < 0
                    or ni >= flat_mask.shape[0]
                    or nj < 0
                    or nj >= flat_mask.shape[1]
                ):
                    continue
                if labels[ni, nj] != labels[ci, cj]:
                    continue
                if flat_mask[ni, nj] >= fmin:
                    continue
                fmin = flat_mask[ni, nj]
                dir = code

            flowdir[ci, cj] = dir

    return flowdir


def _compute_flowdir_total(
    dem: npt.NDArray[np.number],
    directions: D8Directions = D8Directions(),
    step_size: int = 2,
) -> npt.NDArray[np.integer]:
    flowdir = _compute_flowdir_simple(dem, directions=directions)

    is_low_edge, is_high_edge = find_flat_edges(dem, flowdir, directions=directions)

    flat_labels, _, nanfill = label_flats(dem, is_low_edge, directions=directions)

    is_high_edge = is_high_edge & (flat_labels != nanfill)

    flat_gradient_away, flat_height = compute_away_from_high(
        flowdir, flat_labels, is_high_edge, directions=directions
    )

    flat_gradient = compute_towards_low(
        flowdir,
        flat_labels,
        flat_gradient_away,
        is_low_edge,
        flat_height,
        directions=directions,
        step_size=step_size,
    )

    flat_flowdir = compute_masked_flowdir(
        flat_gradient, flat_labels, directions=directions
    )

    flowdir[flowdir == 0] = flat_flowdir[flowdir == 0]
    return flowdir


def compute_flowdir(
    dem: npt.NDArray[np.number],
    directions: D8Directions = D8Directions(),
    resolve_flat: bool = True,
    step_size: int = 2,
) -> npt.NDArray[np.integer]:
    if resolve_flat:
        return _compute_flowdir_total(dem, directions=directions, step_size=step_size)
    else:
        return _compute_flowdir_simple(dem, directions=directions)


def compute_indegree(
    flowdirs: npt.NDArray[np.integer], directions: D8Directions = D8Directions()
) -> npt.NDArray[np.integer]:
    indegree = np.zeros(flowdirs.shape, dtype=np.integer)
    dsi, dsj, _ = compute_downstream_indices(flowdirs, directions=directions)

    for flowdir in np.unique(flowdirs):
        if flowdir == 0:
            continue
        is_Valid_ds = (
            (flowdirs == flowdir)
            & (dsi >= 0)
            & (dsi < flowdirs.shape[0])
            & (dsj >= 0)
            & (dsj < flowdirs.shape[1])
        )
        indegree[dsi[is_Valid_ds], dsj[is_Valid_ds]] += 1

    return indegree


def compute_flowdir_graph(
    flowdirs: npt.NDArray[np.integer],
    valid: npt.NDArray[np.bool] | None = None,
    directions: D8Directions = D8Directions(),
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]]:
    if valid is not None:
        assert (
            valid.shape == flowdirs.shape
        ), f"Shape for FLOWDIR and VALID mask must match, but got valid shape {flowdirs.shape} and flowdir shape {valid.shape} instead"
    else:
        valid = np.full(flowdirs.shape, True, dtype=np.bool)

    i, j = np.meshgrid(
        np.arange(flowdirs.shape[0], dtype=np.integer),
        np.arange(flowdirs.shape[1], dtype=np.integer),
        indexing="ij",
    )
    dsi, dsj, _ = compute_downstream_indices(flowdirs, directions=directions)

    graphi = np.stack(
        (
            i[valid],
            dsi[valid],
            np.full(i[valid].size, np.nan),
        ),
        axis=1,
    ).ravel(order="C")
    graphj = np.stack(
        (
            j[valid],
            dsj[valid],
            np.full(j[valid].size, np.nan),
        ),
        axis=1,
    ).ravel(order="C")
    return graphi, graphj


def compute_accumulation(
    flowdirs: npt.NDArray[np.integer],
    valids=None,
    weights=None,
    indegrees: npt.NDArray[np.integer] | None = None,
    dsij: npt.NDArray[np.integer] | None = None,
    directions: D8Directions = D8Directions(),
) -> np.ndarray:
    # Initialisation
    I, J = flowdirs.shape

    if indegrees is None:
        indegrees = compute_indegree(flowdirs, directions=directions)
    else:
        assert (
            indegrees.shape == flowdirs.shape
        ), f"Shape for flowdir and indegree must match, but got indegree shape {indegrees.shape} and flowdir shape {flowdirs.shape} instead"

    if valids is None:
        valids = (flowdirs != 0) | (indegrees > 0)
    else:
        assert (
            valids.shape == flowdirs.shape
        ), f"Shape for flowidr and valid mask must match, but got valid shape {valids.shape} and flowdir shape {flowdirs.shape} instead"

    if weights is None:
        weights = np.where(valids, 1, 0).astype(np.uint64)
    else:
        assert (
            weights.shape == flowdirs.shape
        ), f"Shape for flowdir and weight must match, but got weight shape {weights.shape} and flowdir shape {flowdirs.shape} instead"
        weights = np.where(valids, weights, 0)

    if dsij is None:
        _, _, dsij = compute_downstream_indices(flowdirs, directions=directions)
    else:
        assert (
            dsij.shape == flowdirs.shape
        ), f"Shape for flowdir and downstream ij indices must match, but got dsij: {dsij.shape} and flowdir: {flowdirs.shape} instead"

    indegrees = indegrees.flatten(order="F")
    valids = valids.flatten(order="F")
    weights = weights.flatten(order="F")
    dsij = dsij.flatten(order="F")
    flowdirs = flowdirs.flatten(order="F")

    # Initialize accumulation with self weight
    accumulation = weights.ravel().astype(weights.dtype, copy=True)

    # Queue sources (indeg == 0) among valid cells
    q = deque(np.flatnonzero((indegrees == 0) & valids))

    # Topological propagation
    while q:
        u = q.popleft()
        v = dsij[u]
        if not valids[v]:
            continue
        accumulation[v] += accumulation[u]
        indegrees[v] -= 1
        if indegrees[v] == 0:
            q.append(v)

    accumulation = accumulation.reshape(I, J, order="F")

    return accumulation
