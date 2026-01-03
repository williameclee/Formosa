from collections import deque
import numpy as np

from .d8directions import D8Directions
from formosa.flow_distance_loop import _flow_distance_loop
from formosa.towards_low_loop import _towards_low_loop
# C-extension
from formosa.geomorphology.away_from_high_loop import away_from_high_loop

# from tqdm import tqdm
import numpy.typing as npt


def _compute_flowdir_simple(
    dem: npt.NDArray[np.number],
    directions: D8Directions = D8Directions(),
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.bool]]:
    is_low_flat = find_flat(dem, directions=directions)
    flat_neighbours, _, _ = get_neighbour_values(
        is_low_flat, directions=D8Directions(window=3), include_self=False
    )
    is_1px_flat = is_low_flat & ~np.any(flat_neighbours, axis=0)

    neighbours, codes, _ = get_neighbour_values(
        dem, directions=directions, include_self=True, pad_value=np.max(dem) + 1
    )
    flow2self_code = np.where(np.all(directions.offsets == [0, 0], axis=1))[0][0]
    flowdir = np.full(dem.shape, flow2self_code, dtype=np.int32)
    # find where not all neighbours are nan
    valid_mask = ~np.all(np.isnan(neighbours), axis=0)
    flowdir[valid_mask] = np.nanargmin(neighbours[:, valid_mask], axis=0)
    # flowdir = np.nanargmin(neighbours, axis=0)

    is_ambiguous = find_ambiguous(dem, directions=directions)
    is_ambiguous = is_ambiguous & ~is_1px_flat

    flowdir = codes[flowdir].astype(np.int32)
    is_flat = flowdir == 0
    return flowdir, is_flat


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
) -> tuple[npt.NDArray[np.int32], int, int]:
    """
    From R. Barnes et al., 2014, Algorithm 4 (p. 133)
    """
    ## Input validation and initialisation
    if isinstance(low_edges, np.ndarray):
        assert (
            dem.shape == low_edges.shape
        ), f"DEM and LOW_EDGES must have the same shape, got {dem.shape} and {low_edges.shape} instead"

        ii, jj = np.indices(dem.shape, dtype=np.uint16)
        low_edges = deque(zip(ii[low_edges], jj[low_edges]))  # type: ignore TODO: figure out what the type error actually is

    ## Initialisation
    I, J = dem.shape
    nanfill = 0
    labels: npt.NDArray[np.int32] = np.full(dem.shape, nanfill)
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
    offsets = np.zeros((len(directions.codes), 2), dtype=np.int16)
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
    valids: npt.NDArray[np.bool] | None = None,
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer], npt.NDArray[np.int32]]:
    if valids is None:
        valids = ~np.isnan(flowdirs)
    elif isinstance(valids, np.ndarray):
        assert (
            valids.shape == flowdirs.shape
        ), f"Shapes for flow direction ({flowdirs.shape}) and valid mask ({valids.shape}) do not match."
    else:
        raise TypeError(
            f"Expected valids to be None or np.ndarray, got {type(valids)} instead."
        )

    I, J = flowdirs.shape
    ii, jj = np.meshgrid(
        np.arange(I, dtype=np.int32), np.arange(J, dtype=np.int32), indexing="ij"
    )
    di, dj = directions.code2d8offset(flowdirs)
    dsi = (ii.astype(np.int16) + (di).astype(np.int16)).astype(np.int16)
    dsj = (jj.astype(np.int16) + (dj).astype(np.int16)).astype(np.int16)
    dsij: npt.NDArray[np.int32] = dsj.astype(np.int32) * I + dsi.astype(np.int32)

    if np.any((dsi < 0) | (dsi >= I) | (dsj < 0) | (dsj >= J)):
        raise ValueError("Some downstream indices out of bounds")

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
    high_edges: npt.NDArray[np.bool] | deque[tuple[int, int]] | list[tuple[int, int]],
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

        ii, jj = np.indices(flowdirs.shape, dtype=np.int32)
        high_edges = list(zip(ii[high_edges], jj[high_edges]))  # type: ignore TODO: figure out what the type error actually is
    elif isinstance(high_edges, deque):
        high_edges = list(high_edges)

    shape_ij: tuple[int, int] = flowdirs.shape
    flat_mask = np.zeros(shape_ij, dtype=np.int32)
    flat_height = np.zeros((np.nanmax(labels) + 1,), dtype=np.int32)

    flat_mask, flat_height = away_from_high_loop(
        flowdirs.astype(np.int32, copy=False),
        flat_mask.astype(np.int32, copy=False),
        flat_height.astype(np.int32, copy=False),
        labels.astype(np.int32, copy=False),
        high_edges,
        directions.offsets.astype(np.int32),
    )
    return flat_mask, flat_height


def compute_towards_low(
    flowdirs: npt.NDArray[np.number],
    labels: npt.NDArray[np.number],
    flat_mask: npt.NDArray[np.integer],
    low_edges: npt.NDArray[np.bool] | deque[tuple[int, int]] | list[tuple[int, int]],
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

        ii, jj = np.indices(flowdirs.shape, dtype=np.int32)
        low_edges = list(zip(ii[low_edges], jj[low_edges]))  # type: ignore TODO: figure out what the type error actually is
    elif isinstance(low_edges, deque):
        low_edges = list(low_edges)

    flat_mask = -flat_mask
    flat_height = np.zeros((np.nanmax(labels) + 1,), dtype=labels.dtype)

    flat_mask = _towards_low_loop(
        flowdirs.astype(np.int32, copy=False),
        flat_mask.astype(np.int32, copy=False),
        flat_height.astype(np.int32, copy=False),
        labels.astype(np.int32, copy=False),
        low_edges,
        directions.offsets.astype(np.int32),
        flowdirs.shape,
        step_size,
    )
    return flat_mask


def compute_masked_flowdir(
    flat_mask, labels, directions: D8Directions = D8Directions()
):
    """
    From R. Barnes et al., 2014, Algorithm 7 (p. 134)
    """

    flowdir = np.zeros(flat_mask.shape, dtype=np.int32)

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
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.bool], npt.NDArray[np.integer]]:
    flowdir, is_flat = _compute_flowdir_simple(dem, directions=directions)

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
    return flowdir, is_flat, flat_gradient


def compute_flowdir(
    dem: npt.NDArray[np.number],
    directions: D8Directions = D8Directions(),
    resolve_flat: bool = True,
    step_size: int = 4,
) -> tuple[
    npt.NDArray[np.integer], npt.NDArray[np.bool], npt.NDArray[np.integer] | None
]:
    if resolve_flat:
        flowdir, is_flat, flat_gradient = _compute_flowdir_total(
            dem, directions=directions, step_size=step_size
        )
    else:
        flowdir, is_flat = _compute_flowdir_simple(dem, directions=directions)
        flat_gradient = None
    return flowdir, is_flat, flat_gradient


def compute_indegree(
    flowdirs: npt.NDArray[np.integer], directions: D8Directions = D8Directions()
) -> npt.NDArray[np.integer]:
    indegree = np.zeros(flowdirs.shape, dtype=np.int32)
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
    x: npt.NDArray[np.number] | None = None,
    y: npt.NDArray[np.number] | None = None,
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]]:
    if valid is not None:
        assert (
            valid.shape == flowdirs.shape
        ), f"Shape for FLOWDIR and VALID mask must match, but got valid shape {flowdirs.shape} and flowdir shape {valid.shape} instead"
    else:
        valid = np.full(flowdirs.shape, True, dtype=np.bool)

    i, j = np.meshgrid(
        np.arange(flowdirs.shape[0], dtype=np.int32),
        np.arange(flowdirs.shape[1], dtype=np.int32),
        indexing="ij",
    )
    dsi, dsj, _ = compute_downstream_indices(flowdirs, directions=directions)

    if x is not None and y is not None:
        j, i = x, y

        # Map i,j to actual coordinates
        dsj, dsi = x[dsi, dsj], y[dsi, dsj]

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


def compute_strahler_order(
    flowdir: npt.NDArray[np.integer] | None = None,
    directions: D8Directions = D8Directions(),
    indegrees: npt.NDArray[np.integer] | None = None,
    downstream_ij: npt.NDArray[np.integer] | None = None,
) -> npt.NDArray[np.integer]:
    if flowdir is None and (indegrees is None or downstream_ij is None):
        raise ValueError(
            "[FORMOSA] Either FLOWDIR or (INDEGREES and DOWNSTREAM_IJ) must be provided"
        )
    elif (indegrees is None or downstream_ij is None) and flowdir is not None:
        downstream_i, downstreamj, _ = compute_downstream_indices(
            flowdir, directions=directions
        )
        indegrees = compute_indegree(flowdir, directions=directions)
    else:
        raise NotImplementedError("Unknown case for FLOWDIR and INDEGREES")

    strahler_order = np.zeros(indegrees.shape, dtype=np.int32)
    strahler_order[indegrees == 0] = 1

    ii, jj = np.indices(indegrees.shape, dtype=np.int32)
    seeds = deque(zip(ii[indegrees == 0], jj[indegrees == 0]))  # type: ignore TODO: figure out what the type error actually is

    while seeds:
        ci, cj = seeds.popleft()
        dsi, dsj = (
            downstream_i[ci, cj],
            downstreamj[ci, cj],
        )
        if (ci, cj) == (dsi, dsj):
            continue
        if strahler_order[dsi, dsj] < strahler_order[ci, cj]:
            strahler_order[dsi, dsj] = strahler_order[ci, cj]
        else:
            strahler_order[dsi, dsj] += 1
        indegrees[dsi, dsj] -= 1
        if indegrees[dsi, dsj] == 0:
            seeds.append((dsi, dsj))
    return strahler_order


def compute_flow_distance(
    flowdir: npt.NDArray[np.integer], directions: D8Directions = D8Directions()
) -> npt.NDArray[np.integer]:
    downstream_i, downstreamj, _ = compute_downstream_indices(
        flowdir, directions=directions
    )

    distance: npt.NDArray[np.int32] = np.zeros(flowdir.shape, dtype=np.int32)

    ii, jj = np.indices(flowdir.shape, dtype=np.int32)
    seeds: list[tuple[int, int]] = list(zip(ii[flowdir == 0], jj[flowdir == 0]))  # type: ignore TODO: figure out what the type error actually is
    distance[flowdir == 0] = 1

    shape_ij: tuple[int, int] = flowdir.shape

    distance = _flow_distance_loop(
        distance,
        seeds,
        directions.offsets.astype(np.int32, copy=False),
        downstream_i.astype(np.int32, copy=False),
        downstreamj.astype(np.int32, copy=False),
        shape_ij,
    )
    return distance


def label_watersheds(
    flowdir: npt.NDArray[np.integer],
    directions: D8Directions = D8Directions(),
    valids: npt.NDArray[np.bool] | None = None,
) -> npt.NDArray[np.int32]:
    """
    Find and label watersheds in a DEM based on flow direction.
    """
    if valids is None:
        valids = ~np.isnan(flowdir)
    elif isinstance(valids, np.ndarray):
        assert (
            valids.shape == flowdir.shape
        ), f"Shape for flow direction ({valids.shape}) and valid mask ({flowdir.shape}) do not match."
        valids = valids.astype(np.bool, copy=False) & (~np.isnan(flowdir))
        flowdir = np.where(valids, flowdir, np.nan)
    else:
        raise TypeError(
            f"[FORMOSA] VALIDS must be either None or a numpy array, got {type(valids)} instead."
        )

    I, J = flowdir.shape
    ii, jj = np.meshgrid(
        np.arange(I, dtype=np.int32), np.arange(J, dtype=np.int32), indexing="ij"
    )
    codes: list[int] = directions.codes.tolist()
    offsets: list[tuple[int, int]] = [
        (int(di), int(dj)) for di, dj in directions.offsets.astype(np.int32, copy=False)
    ]

    seeds: list[tuple[int, int]] = list(
        zip(ii[valids & (flowdir == 0)], jj[valids & (flowdir == 0)])
    )

    watershed = -np.ones(flowdir.shape, dtype=np.int32)

    for label, seed in enumerate(seeds):
        to_fill: list[tuple[int, int]] = [seed]

        while to_fill:
            ci, cj = to_fill.pop(0)
            watershed[ci, cj] = label
            for code, (di, dj) in zip(codes, offsets):
                ni, nj = ci - di, cj - dj
                if (ni < 0 or ni >= I) or (nj < 0 or nj >= J):
                    continue
                elif not valids[ni, nj]:
                    continue
                elif watershed[ni, nj] != -1:
                    continue

                if flowdir[ni, nj] == code:
                    to_fill.append((ni, nj))
    watershed = watershed + 1  # make background 0 and watersheds start from 1
    return watershed


def compute_back_distance(
    flowdir: npt.NDArray[np.integer],
    directions: D8Directions = D8Directions(),
    valids: npt.NDArray[np.bool] | None = None,
) -> npt.NDArray[np.float32]:
    if valids is None:
        valids = ~np.isnan(flowdir)
    elif isinstance(valids, np.ndarray):
        assert (
            valids.shape == flowdir.shape
        ), f"Shape for flow direction ({valids.shape}) and valid mask ({flowdir.shape}) do not match."
        valids = valids.astype(np.bool, copy=False) & (~np.isnan(flowdir))
        flowdir = np.where(valids, flowdir, np.nan)
    else:
        raise TypeError(
            f"[FORMOSA] VALIDS must be either None or a numpy array, got {type(valids)} instead."
        )

    I, J = flowdir.shape
    ii, jj = np.meshgrid(
        np.arange(I, dtype=np.int32), np.arange(J, dtype=np.int32), indexing="ij"
    )
    codes: list[int] = directions.codes.tolist()
    offsets: list[tuple[int, int]] = [
        (int(di), int(dj)) for di, dj in directions.offsets.astype(np.int32, copy=False)
    ]
    lengths: list[float] = [float(np.hypot(di, dj)) for di, dj in offsets]

    seeds: list[tuple[int, int]] = list(
        zip(ii[valids & (flowdir == 0)], jj[valids & (flowdir == 0)])
    )

    distance = -np.ones(flowdir.shape, dtype=np.int32)

    for si, sj in seeds:
        distance[si, sj] = 0

    for seed in seeds:
        to_fill: list[tuple[int, int]] = [seed]

        while to_fill:
            ci, cj = to_fill.pop(0)

            for code, (di, dj) in zip(codes, offsets):
                ni, nj = ci - di, cj - dj
                if (ni < 0 or ni >= I) or (nj < 0 or nj >= J):
                    continue
                elif not valids[ni, nj]:
                    continue
                elif distance[ni, nj] != -1:
                    continue
                elif flowdir[ni, nj] != code:
                    continue

                distance[ni, nj] = distance[ci, cj] + lengths[codes.index(code)]
                to_fill.append((ni, nj))
    return distance.astype(np.float32, copy=False)
