# Last modified
#   2026-06-11, En-Chi Lee (williameclee@gmail.com)
#     - Updated function and argument names to match the
#       standardised names.
#   2026-07-01, En-Chi Lee (williameclee@gmail.com)
#     - Added test cases for `compute_downstream_indices`,
#       `create_flowgraph`, and `compute_flow_strahler_order`.
#   2026-07-02, En-Chi Lee (williameclee@gmail.com)
#     - Added test case for `compute_downstream_indices` with
#       validity mask.
#   2026-07-09, En-Chi Lee (williameclee@gmail.com)
#     - Added test case for the Fortran implementation of
#       `construct_flowgraph`.
#   2026-07-12, En-Chi Lee (williameclee@gmail.com)
#     - Added test cases for function `test_locate_invalid_graph_topogtaphy`.
#   2026-07-14, En-Chi Lee (williameclee@gmail.com)
#     - Updated `geomorphology.flowdir` to the new submodule name.
#   2026-07-29, En-Chi Lee (williameclee@gmail.com)
#     - Added test cases for function `simplify_flowgraph`.
#     - Added complete topology-intersection scan-and-retry
#       regression tests.
#   2026-07-30, En-Chi Lee (williameclee@gmail.com)
#     - Various minor refactors and type annotation enhancements.
#   2026-07-31, En-Chi Lee (williameclee@gmail.com)
#     - Updated tests to match the updated `simplify_flowgraph`
#       interface; also added additional tests for the new interface.
#   2026-08-03, En-Chi Lee (williameclee@gmail.com)
#     - Added test cases for function `find_acyclic_flowdirs`.
#   2026-08-04, En-Chi Lee (williameclee@gmail.com)
#     - Added test cases for FORTRAN error code handling.
#   2026-08-05, En-Chi Lee (williameclee@gmail.com)
#     - Added a regression check for nonstandard old-style FORTRAN
#       kind declarations.
#   2026-08-07 [PR 33], En-Chi Lee (williameclee@gmail.com)
#     - Added tests for the priority-flood depression-filling
#       algorithm.
#     - Added tests for the ocean basin masking algorithms 
#       (`label_mask_areas`).

import re
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

from formosa import D8Directions
import formosa.geomorphology.flowdir as flowdir
from formosa.geomorphology.flowdir.raster import raster as raster_module
from formosa.geomorphology.flowdir.graphs import graphs as graphs_module
from formosa.geomorphology.flowdir_f import flowdir_graphs as graphs_f
from formosa.geomorphology.flowdir_f import flowdir_raster as raster_f
from formosa.geomorphology.flowdir_f import utils as utils_f

T = True
F = False


def test_detect_ocean_basins_labels_separate_boundary_components():
    dem = np.full((7, 9), 5.0, dtype=np.float32)
    dem[0:3, 0:2] = 0.0
    dem[4:7, 7:9] = -1.0
    dem[3, 4] = 0.0  # Enclosed low cell is not an ocean basin.

    basins = flowdir.detect_ocean_basins_from_boundary(dem)

    left_label = basins[0, 0]
    right_label = basins[-1, -1]
    assert left_label > 0
    assert right_label > 0
    assert left_label != right_label
    assert np.all(basins[0:3, 0:2] == left_label)
    assert np.all(basins[4:7, 7:9] == right_label)
    assert basins[3, 4] == 0
    assert np.all(basins[dem == 5.0] == 0)


def test_detect_ocean_basins_respects_valids_and_exact_level_mode():
    dem = np.array(
        [[0.0, -1.0, -1.0, 5.0], [0.0, -1.0, -1.0, 5.0]],
        dtype=np.float32,
    )
    valids = np.ones(dem.shape, dtype=bool)
    valids[1, 0] = False

    exact = flowdir.detect_ocean_basins_from_boundary(
        dem, valids=valids, flood_below=False
    )
    flooded = flowdir.detect_ocean_basins_from_boundary(
        dem, valids=valids, flood_below=True
    )

    assert exact[0, 0] > 0
    assert not np.any(exact[:, 1:3])
    assert exact[1, 0] == 0
    assert np.all(flooded[:, 1:3] > 0)
    assert flooded[1, 0] == 0


def test_invalidate_ocean_basins_filters_by_inclusive_size():
    dem = np.full((7, 9), 5.0, dtype=np.float32)
    dem[0:2, 0:3] = 0.0  # Six-cell boundary basin.
    dem[5:7, 7:9] = 0.0  # Four-cell boundary basin.
    dem[3, 3:6] = 0.0  # Enclosed basin is never an ocean candidate.
    valids = np.ones(dem.shape, dtype=bool)
    valids[6, 0] = False

    result = flowdir.invalidate_ocean_basins(dem, valids=valids, min_size=6)

    assert not np.any(result[0:2, 0:3])
    assert np.all(result[5:7, 7:9])
    assert np.all(result[3, 3:6])
    assert not result[6, 0]
    np.testing.assert_array_equal(valids[0:2, 0:3], np.ones((2, 3), dtype=bool))


@pytest.mark.parametrize("minimum_basin_size", [0, -1])
def test_invalidate_ocean_basins_rejects_nonpositive_size(minimum_basin_size):
    with pytest.raises(ValueError, match="at least 1"):
        flowdir.invalidate_ocean_basins(
            np.zeros((2, 2), dtype=np.float32),
            min_size=minimum_basin_size,
        )


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (6, 1), (2, 8), (9, 2)])
def test_detect_ocean_basins_handles_boundary_only_grids(shape):
    ocean = np.zeros(shape, dtype=np.float32)
    land = np.ones(shape, dtype=np.float32)

    ocean_basins = flowdir.detect_ocean_basins_from_boundary(ocean)
    land_basins = flowdir.detect_ocean_basins_from_boundary(land)

    assert np.all(ocean_basins == ocean_basins.flat[0])
    assert ocean_basins.flat[0] > 0
    assert not np.any(land_basins)


def test_detect_ocean_basins_respects_connectivity_scheme():
    dem = np.full((3, 3), 5.0, dtype=np.float32)
    dem[0, 0] = 0.0
    dem[1, 1] = 0.0
    d4 = SimpleNamespace(
        offsets=np.array([[-1, 0], [0, -1], [0, 1], [1, 0]], dtype=np.int32)
    )

    d8_basins = flowdir.detect_ocean_basins_from_boundary(dem)
    d4_basins = flowdir.detect_ocean_basins_from_boundary(dem, dir_scheme=d4)  # type: ignore

    assert d8_basins[1, 1] == d8_basins[0, 0]
    assert d4_basins[0, 0] > 0
    assert d4_basins[1, 1] == 0


def test_detect_ocean_basins_excludes_nonfinite_and_all_invalid_cells():
    dem = np.array([[0.0, np.nan], [np.inf, -np.inf]], dtype=np.float32)
    basins = flowdir.detect_ocean_basins_from_boundary(dem)
    all_invalid = flowdir.detect_ocean_basins_from_boundary(
        np.zeros((2, 3), dtype=np.float32),
        valids=np.zeros((2, 3), dtype=bool),
    )

    assert basins[0, 0] > 0
    assert not np.any(basins[~np.isfinite(dem)])
    assert not np.any(all_invalid)


@pytest.mark.parametrize("dtype", [np.int16, np.float32, np.float64])
def test_detect_ocean_basins_accepts_numeric_dtypes_and_noncontiguous_views(dtype):
    source = np.zeros((8, 10), dtype=dtype)
    dem = source[::2, ::2]
    original = dem.copy()

    basins = flowdir.detect_ocean_basins_from_boundary(dem)

    assert basins.shape == dem.shape
    assert basins.dtype == np.int32
    assert np.all(basins > 0)
    np.testing.assert_array_equal(dem, original)


@pytest.mark.parametrize("min_size", [True, 1.5, "2"])
def test_invalidate_ocean_basins_rejects_noninteger_size(min_size):
    with pytest.raises(TypeError, match="integer"):
        flowdir.invalidate_ocean_basins(
            np.zeros((2, 2), dtype=np.float32), min_size=min_size
        )


@pytest.mark.parametrize("ocean_level", [np.nan, np.inf, -np.inf])
def test_detect_ocean_basins_rejects_nonfinite_ocean_level(ocean_level):
    with pytest.raises(ValueError, match="finite"):
        flowdir.detect_ocean_basins_from_boundary(
            np.zeros((2, 2), dtype=np.float32), ocean_level=ocean_level
        )


def test_detect_ocean_basins_rejects_nonboolean_flood_below_and_complex_dem():
    dem = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(TypeError, match="boolean"):
        flowdir.detect_ocean_basins_from_boundary(dem, flood_below="false")  # type: ignore
    with pytest.raises(TypeError, match="real-valued"):
        flowdir.detect_ocean_basins_from_boundary(dem.astype(np.complex64))


def test_fill_depressions():
    dem = np.array(
        [[5.0, 5.0, 5.0], [5.0, 1.0, 5.0], [5.0, 5.0, 5.0]],
        dtype=np.float64,
    )

    filled = flowdir.fill_depressions(dem)

    np.testing.assert_array_equal(filled, np.full((3, 3), 5.0))
    assert filled.dtype == dem.dtype
    assert dem[1, 1] == 1.0


@pytest.mark.parametrize(
    ("dem", "expected"),
    [
        (
            [
                [9, 9, 9, 9, 9],
                [9, 3, 3, 3, 9],
                [9, 3, 1, 3, 4],
                [9, 3, 3, 3, 9],
                [9, 9, 9, 9, 9],
            ],
            [
                [9, 9, 9, 9, 9],
                [9, 4, 4, 4, 9],
                [9, 4, 4, 4, 4],
                [9, 4, 4, 4, 9],
                [9, 9, 9, 9, 9],
            ],
        ),
        (
            [[0, 0, 0, 0], [4, -3, -2, 0], [4, -4, -1, 0], [4, 4, 4, 4]],
            [[0, 0, 0, 0], [4, 0, 0, 0], [4, 0, 0, 0], [4, 4, 4, 4]],
        ),
        (
            [[5, 5, 5, 5], [5, 2, 2, 1], [5, 2, 0, 5], [5, 5, 5, 5]],
            [[5, 5, 5, 5], [5, 2, 2, 1], [5, 2, 1, 5], [5, 5, 5, 5]],
        ),
    ],
    ids=["single-spill-basin", "negative-basin", "open-channel"],
)
def test_fill_depressions_fortran_reference_terrains(dem, expected):
    filled = flowdir.fill_depressions(np.asarray(dem, dtype=np.float32))

    np.testing.assert_array_equal(filled, np.asarray(expected, dtype=np.float32))


@pytest.mark.parametrize("dtype", [np.int16, np.float32, np.float64])
def test_fill_depressions_preserves_shape_dtype_and_input(dtype):
    source = np.array(
        [[7, 7, 7, 7], [7, 1, 2, 7], [7, 3, 0, 7], [7, 7, 7, 7]],
        dtype=dtype,
    )
    dem = source[:, ::-1]
    original = dem.copy()

    filled = flowdir.fill_depressions(dem)

    assert filled.shape == dem.shape
    assert filled.dtype == dem.dtype
    np.testing.assert_array_equal(dem, original)
    np.testing.assert_array_equal(filled, np.full(dem.shape, 7, dtype=dtype))


@pytest.mark.parametrize("shape", [(1, 7), (6, 1), (2, 8), (9, 2)])
def test_fill_depressions_boundary_only_grids_are_unchanged(shape):
    dem = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

    filled = flowdir.fill_depressions(dem)

    np.testing.assert_array_equal(filled, dem)


def test_fill_depressions_is_monotonic_and_idempotent_randomly():
    rng = np.random.default_rng(8675309)
    for _ in range(100):
        shape = tuple(rng.integers(3, 20, size=2))
        dem = rng.integers(-1000, 1000, size=shape).astype(np.float32)

        filled = flowdir.fill_depressions(dem)
        filled_twice = flowdir.fill_depressions(filled)

        assert np.all(filled >= dem)
        np.testing.assert_array_equal(filled[0, :], dem[0, :])
        np.testing.assert_array_equal(filled[-1, :], dem[-1, :])
        np.testing.assert_array_equal(filled[:, 0], dem[:, 0])
        np.testing.assert_array_equal(filled[:, -1], dem[:, -1])
        np.testing.assert_array_equal(filled_twice, filled)


def test_fill_depressions_validates_mask_shape():
    with pytest.raises(ValueError, match="must have the same shape"):
        flowdir.fill_depressions(
            np.ones((3, 3), dtype=np.float32),
            valids=np.ones((2, 2), dtype=bool),
        )


def test_fill_depressions_fortran_preserves_invalid_cells():
    dem = np.array(
        [[5.0, 5.0, 5.0], [5.0, -9999.0, 1.0], [5.0, 5.0, 5.0]],
        dtype=np.float32,
    )
    valids = np.ones(dem.shape, dtype=bool)
    valids[1, 1] = False

    filled = flowdir.fill_depressions(dem, valids=valids)

    assert filled[1, 1] == dem[1, 1]


def test_fill_depressions_treats_internal_invalids_as_outlets():
    dem = np.full((5, 5), 5.0, dtype=np.float32)
    dem[1:4, 1:4] = 1.0
    dem[2, 2] = -9999.0
    valids = np.ones(dem.shape, dtype=bool)
    valids[2, 2] = False

    filled = flowdir.fill_depressions(dem, valids=valids)

    np.testing.assert_array_equal(filled, dem)


def test_fill_depressions_treats_boundary_invalids_as_outlets():
    dem = np.full((5, 5), -9999.0, dtype=np.float32)
    dem[1:4, 1:4] = 5.0
    dem[2, 2] = 1.0
    valids = np.zeros(dem.shape, dtype=bool)
    valids[1:4, 1:4] = True

    filled = flowdir.fill_depressions(dem, valids=valids)

    np.testing.assert_array_equal(filled[valids], np.full(valids.sum(), 5.0))
    np.testing.assert_array_equal(filled[~valids], dem[~valids])


def test_fill_depressions_uses_diagonal_invalid_adjacency():
    dem = np.full((5, 5), 10.0, dtype=np.float32)
    dem[1, 1] = 1.0
    dem[2, 2] = -9999.0
    valids = np.ones(dem.shape, dtype=bool)
    valids[2, 2] = False

    filled = flowdir.fill_depressions(dem, valids=valids)

    assert filled[1, 1] == 1.0
    assert filled[2, 2] == -9999.0


def test_fill_depressions_deduplicates_outlets_adjacent_to_many_invalids():
    rows, cols = np.indices((9, 9))
    valids = (rows + cols) % 2 == 0
    dem = np.arange(81, dtype=np.float32).reshape(9, 9)
    dem[~valids] = -9999.0

    filled = flowdir.fill_depressions(dem, valids=valids)

    np.testing.assert_array_equal(filled, dem)


def test_fill_depressions_fills_valid_island_surrounded_by_invalids():
    dem = np.full((9, 9), -9999.0, dtype=np.float32)
    dem[2:7, 2:7] = 3.0
    dem[3:6, 3:6] = 1.0
    valids = np.zeros(dem.shape, dtype=bool)
    valids[2:7, 2:7] = True

    filled = flowdir.fill_depressions(dem, valids=valids)

    np.testing.assert_array_equal(filled[valids], np.full(valids.sum(), 3.0))
    np.testing.assert_array_equal(filled[~valids], dem[~valids])


def test_fill_depressions_all_invalid_is_unchanged():
    dem = np.arange(12, dtype=np.float32).reshape(3, 4)

    filled = flowdir.fill_depressions(dem, valids=np.zeros(dem.shape, dtype=bool))

    np.testing.assert_array_equal(filled, dem)
    assert filled is not dem


def test_fill_depressions_processes_multiple_large_basins_together():
    dem = np.full((7, 13), 10.0, dtype=np.float32)
    dem[1:6, 1:6] = 5.0
    dem[1:6, 7:12] = 5.0
    dem[3, 3] = 0.0
    dem[3, 9] = -1.0
    dem[1, 1] = 1.0
    dem[1, 11] = 2.0

    filled = flowdir.fill_depressions(dem, max_fill_size=24)

    expected = dem.copy()
    expected[1, 1] = 5.0
    expected[1, 11] = 5.0
    np.testing.assert_array_equal(filled, expected)


def test_label_mask_areas():
    dir_scheme = D8Directions()
    mask = np.array(
        [
            [T, F, T, F, T],
            [T, F, T, F, T],
            [T, F, T, F, T],
            [T, F, T, F, T],
        ],
        dtype=bool,
    )
    labels, err_code = raster_f.label_mask_areas(mask, dir_scheme.offsets)
    assert err_code == 0
    assert ~np.any(labels[:, 1])
    assert ~np.any(labels[:, 3])
    c1 = np.unique(labels[:, 0])
    assert np.size(c1) == 1
    c2 = np.unique(labels[:, 2])
    assert np.size(c2) == 1
    c3 = np.unique(labels[:, 4])
    assert np.size(c3) == 1
    assert c1 != c2
    assert c1 != c3
    assert c2 != c3

    mask = np.array(
        [
            [T, F, T, F, T],
            [T, F, T, F, T],
            [T, T, T, F, T],
            [T, F, T, F, T],
        ],
        dtype=bool,
    )
    labels, err_code = raster_f.label_mask_areas(mask, dir_scheme.offsets)
    assert err_code == 0
    assert ~np.any(labels[:, 3])
    c1 = np.unique(labels[:, 0])
    assert np.size(c1) == 1
    c2 = np.unique(labels[:, 2])
    assert np.size(c2) == 1
    c3 = np.unique(labels[:, 4])
    assert np.size(c3) == 1
    assert c1 == c2
    assert c1 != c3


def test_all_allocations_check_status():
    source_root = Path(__file__).parents[1] / "src" / "formosa" / "geomorphology"
    unguarded = []
    for source_path in source_root.rglob("*.f95"):
        lines = source_path.read_text().splitlines()
        for line_number, line in enumerate(lines, start=1):
            if not line.lower().lstrip().startswith("allocate ("):
                continue
            statement = line.strip()
            next_index = line_number
            while statement.endswith("&") and next_index < len(lines):
                statement = f"{statement} {lines[next_index].strip()}"
                next_index += 1
            if "stat=" not in statement.lower():
                unguarded.append(
                    f"{source_path.relative_to(source_root)}:{line_number}"
                )
                continue
            stat_var = statement.lower().split("stat=", 1)[1].split(")", 1)[0].strip()
            status_check = f"if ({stat_var} /= 0)"
            following_lines = " ".join(lines[next_index : next_index + 4]).lower()
            if status_check not in following_lines:
                unguarded.append(
                    f"{source_path.relative_to(source_root)}:{line_number}"
                )

    assert unguarded == []


def test_fortran_sources_avoid_old_style_kind_declarations():
    source_root = Path(__file__).parents[1] / "src" / "formosa" / "geomorphology"
    old_style_declaration = re.compile(
        r"^\s*(integer|logical|real|complex)\s*\*\s*\d+", re.IGNORECASE
    )
    violations = []

    for source_path in source_root.rglob("*.f90"):
        for line_number, line in enumerate(
            source_path.read_text().splitlines(), start=1
        ):
            if old_style_declaration.match(line):
                violations.append(
                    f"{source_path.relative_to(source_root)}:{line_number}"
                )

    assert violations == []


def test_mask2ij_returns_output_capacity_error():
    indices, count, err_code = utils_f.mask2ij(
        np.ones((2, 2), dtype=bool, order="F"),
        2,
    )

    assert count == 2
    assert err_code == 3
    assert indices.shape == (2, 2)


def test_checked_linear_cell_ids_reject_invalid_coordinates_and_ids():
    assert utils_f.ij2id_checked(1, 1, 3, 4) == 1
    assert utils_f.ij2id_checked(3, 4, 3, 4) == 12
    assert utils_f.ij2id_checked(0, 1, 3, 4) == 0
    assert utils_f.ij2id_checked(4, 1, 3, 4) == 0
    assert utils_f.ij2id_checked(1, 0, 3, 4) == 0
    assert utils_f.ij2id_checked(1, 5, 3, 4) == 0

    assert utils_f.id2ij_checked(1, 3, 4) == (1, 1, True)
    assert utils_f.id2ij_checked(12, 3, 4) == (3, 4, True)
    assert utils_f.id2ij_checked(0, 3, 4) == (0, 0, False)
    assert utils_f.id2ij_checked(13, 3, 4) == (0, 0, False)


def test_direction_utilities_infer_input_shapes():
    offsets = np.array([[0, 0], [0, 1], [0, -1]], dtype=np.int32, order="F")
    codes = np.array([0, 1, 5], dtype=np.int8)

    assert utils_f.find_noflow_code(offsets, codes) == 0
    assert np.array_equal(utils_f.find_opposite_codes(offsets, codes), [0, 5, 1])

    lookup = utils_f.fill_offset_lookup(offsets, codes)
    assert np.array_equal(lookup[1], [0, 1])
    assert np.array_equal(lookup[5], [0, -1])


def _assert_min_heap(queue, queue_size, elevations):
    for position in range(queue_size):
        left = 2 * position + 1
        right = left + 1
        parent_elevation = elevations[queue[position] - 1]
        if left < queue_size:
            assert parent_elevation <= elevations[queue[left] - 1]
        if right < queue_size:
            assert parent_elevation <= elevations[queue[right] - 1]


def _drain_priority_queue(queue, queue_size, elevations):
    popped_ids = []
    while queue_size.item() > 0:
        popped, err_code = utils_f.pop_priority_queue(
            queue,
            queue_size,
            elevations,
        )
        assert err_code == 0
        popped_ids.append(popped)
        _assert_min_heap(queue, queue_size.item(), elevations)
    return popped_ids


def test_priority_queue_pushes_into_empty_queue_and_pops_in_elevation_order():
    z = np.array([5, 1, 4, 3, 2], dtype=np.float32)
    queue = np.zeros(z.size, dtype=np.int32)
    queue_size = np.array(0, dtype=np.int32)

    for cell_id in range(1, z.size + 1):
        err_code = utils_f.push_priority_queue(queue, queue_size, cell_id, z)
        assert err_code == 0
        _assert_min_heap(queue, queue_size.item(), z)

    assert _drain_priority_queue(queue, queue_size, z) == [2, 5, 4, 3, 1]


def test_priority_queue_matches_sorted_random_elevations():
    rng = np.random.default_rng(20260806)
    z = rng.permutation(257).astype(np.float32)
    insertion_order = rng.permutation(z.size) + 1
    queue = np.zeros(z.size, dtype=np.int32)
    queue_size = np.array(0, dtype=np.int32)

    for cell_id in insertion_order:
        err_code = utils_f.push_priority_queue(queue, queue_size, int(cell_id), z)
        assert err_code == 0

    expected = (np.argsort(z) + 1).tolist()
    assert _drain_priority_queue(queue, queue_size, z) == expected


def test_priority_queue_handles_equal_elevations_and_reports_invalid_operations():
    z = np.array([2, 1, 1, 3], dtype=np.float32)
    queue = np.zeros(z.size, dtype=np.int32)
    queue_size = np.array(0, dtype=np.int32)

    for cell_id in range(1, z.size + 1):
        assert utils_f.push_priority_queue(queue, queue_size, cell_id, z) == 0

    popped_ids = _drain_priority_queue(queue, queue_size, z)
    popped_elevations = z[np.asarray(popped_ids) - 1]
    assert np.all(popped_elevations[:-1] <= popped_elevations[1:])

    popped, err_code = utils_f.pop_priority_queue(queue, queue_size, z)
    assert popped == 0
    assert err_code == 1

    one_cell_queue = np.zeros(1, dtype=np.int32)
    one_cell_size = np.array(0, dtype=np.int32)
    assert utils_f.push_priority_queue(one_cell_queue, one_cell_size, 1, z) == 0
    assert utils_f.push_priority_queue(one_cell_queue, one_cell_size, 2, z) == 3
    assert one_cell_size.item() == 1
    assert one_cell_queue[0] == 1

    invalid_size = np.array(-1, dtype=np.int32)
    assert utils_f.push_priority_queue(one_cell_queue, invalid_size, 1, z) == 1


def test_flat_synthetic_gradients_follow_breadth_first_layers():
    labels = np.ones((5, 5), dtype=np.int32, order="F")
    offsets = D8Directions().offsets.astype(np.int32, order="F")
    centre = np.zeros(labels.shape, dtype=bool, order="F")
    centre[2, 2] = True

    pulling, err_code = raster_f.create_pulling_syn_grad(labels, centre, offsets)
    assert err_code == 0
    np.testing.assert_array_equal(
        pulling,
        np.array(
            [
                [3, 3, 3, 3, 3],
                [3, 2, 2, 2, 3],
                [3, 2, 1, 2, 3],
                [3, 2, 2, 2, 3],
                [3, 3, 3, 3, 3],
            ],
            dtype=np.int32,
        ),
    )

    pushing, err_code = raster_f.create_pushing_syn_grad(labels, centre, offsets)
    assert err_code == 0
    np.testing.assert_array_equal(pushing, 4 - pulling)


def test_flat_synthetic_gradients_handle_empty_inputs():
    labels = np.zeros((2, 3), dtype=np.int32, order="F")
    edges = np.zeros(labels.shape, dtype=bool, order="F")
    offsets = D8Directions().offsets.astype(np.int32, order="F")

    pushing, pushing_err = raster_f.create_pushing_syn_grad(labels, edges, offsets)
    pulling, pulling_err = raster_f.create_pulling_syn_grad(labels, edges, offsets)

    assert pushing_err == 0
    assert pulling_err == 0
    np.testing.assert_array_equal(pushing, 0)
    np.testing.assert_array_equal(pulling, 0)


def test_downstreamid_3x3():
    dir_scheme = D8Directions(transform_codes=lambda x: x)

    # Config 1
    dirs = np.array(
        [
            [3, 3, 3],
            [3, 3, 3],
            [1, 1, 0],
        ]
    )

    expected_dsi = np.array(
        [
            [1, 1, 1],
            [2, 2, 2],
            [2, 2, 2],
        ]
    )
    expected_dsj = np.array(
        [
            [0, 1, 2],
            [0, 1, 2],
            [1, 2, 2],
        ]
    )
    expected_dsij = np.array(
        [
            [1, 4, 7],
            [2, 5, 8],
            [5, 8, 8],
        ]
    )
    expected_inbounds = np.array(
        [
            [T, T, T],
            [T, T, T],
            [T, T, T],
        ]
    )

    dsi, dsj, dsij, ds_inbounds = flowdir.compute_downstream_indices(
        dirs, dir_scheme=dir_scheme
    )

    np.testing.assert_array_equal(dsi, expected_dsi)
    np.testing.assert_array_equal(dsj, expected_dsj)
    np.testing.assert_array_equal(dsij, expected_dsij)
    np.testing.assert_array_equal(ds_inbounds, expected_inbounds)

    # Config 2
    dirs = np.array(
        [
            [5, 1, 1],
            [5, 1, 1],
            [5, 1, 1],
        ]
    )

    expected_dsi = np.array(
        [
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
        ]
    )
    expected_dsj = np.array(
        [
            [-1, 2, 3],
            [-1, 2, 3],
            [-1, 2, 3],
        ]
    )
    expected_dsij = np.array(
        [
            [-3, 6, 9],
            [-2, 7, 10],
            [-1, 8, 11],
        ]
    )
    expected_inbounds = np.array(
        [
            [F, T, F],
            [F, T, F],
            [F, T, F],
        ]
    )

    with pytest.warns(UserWarning):
        dsi, dsj, dsij, ds_inbounds = flowdir.compute_downstream_indices(
            dirs, dir_scheme=dir_scheme, check=False
        )

    np.testing.assert_array_equal(dsi, expected_dsi)
    np.testing.assert_array_equal(dsj, expected_dsj)
    np.testing.assert_array_equal(dsij, expected_dsij)
    np.testing.assert_array_equal(ds_inbounds, expected_inbounds)

    # Config 4 - with validity mask
    dirs = np.array(
        [
            [3, 3, 3],
            [3, 3, 3],
            [1, 1, 0],
        ]
    )
    valids = np.array([[F, T, T], [T, T, T], [T, T, T]])

    expected_dsi = np.array(
        [
            [-1, 1, 1],
            [2, 2, 2],
            [2, 2, 2],
        ]
    )
    expected_dsj = np.array(
        [
            [-1, 1, 2],
            [0, 1, 2],
            [1, 2, 2],
        ]
    )
    expected_dsij = np.array(
        [
            [-1, 4, 7],
            [2, 5, 8],
            [5, 8, 8],
        ]
    )
    expected_inbounds = np.array(
        [
            [T, T, T],
            [T, T, T],
            [T, T, T],
        ]
    )

    dsi, dsj, dsij, ds_inbounds = flowdir.compute_downstream_indices(
        dirs, dir_scheme=dir_scheme, valids=valids
    )

    np.testing.assert_array_equal(dsi, expected_dsi)
    np.testing.assert_array_equal(dsj, expected_dsj)
    np.testing.assert_array_equal(dsij, expected_dsij)
    np.testing.assert_array_equal(ds_inbounds, expected_inbounds)


def test_downstreamid_4x4():
    dir_scheme = D8Directions(transform_codes=lambda x: x)

    # Config 1
    dirs = np.array(
        [
            [1, 2, 2, 2],
            [8, 1, 1, 1],
            [8, 8, 8, 8],
            [1, 2, 1, 2],
        ]
    )
    expected_dsi = np.array(
        [
            [0, 1, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1],
            [3, 4, 3, 4],
        ]
    )
    expected_dsj = np.array(
        [
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
        ]
    )
    expected_valids = np.array(
        [
            [T, T, T, F],
            [T, T, T, F],
            [T, T, T, F],
            [T, F, T, F],
        ]
    )
    with pytest.warns(UserWarning):
        dsi, dsj, _, ds_valids = flowdir.compute_downstream_indices(
            dirs, dir_scheme=dir_scheme, check=F
        )
    np.testing.assert_array_equal(dsi, expected_dsi)
    np.testing.assert_array_equal(dsj, expected_dsj)
    np.testing.assert_array_equal(ds_valids, expected_valids)


def test_flowgraph_3x3():
    dir_scheme = D8Directions(transform_codes=lambda x: x)

    # Config 1
    dirs = np.array([[3, 3, 3], [3, 3, 3], [1, 1, 0]], dtype=np.uint8, order="F")

    expected_fg_i = np.array(
        [
            *(0, 1, np.nan, 0, 1, np.nan, 0, 1, np.nan),
            *(1, 2, np.nan, 1, 2, np.nan, 1, 2, np.nan),
            *(2, 2, np.nan, 2, 2, np.nan, 2, 2, np.nan),
        ]
    )
    expected_fg_j = np.array(
        [
            *(0, 0, np.nan, 1, 1, np.nan, 2, 2, np.nan),
            *(0, 0, np.nan, 1, 1, np.nan, 2, 2, np.nan),
            *(0, 1, np.nan, 1, 2, np.nan, 2, 2, np.nan),
        ]
    )
    fg_i, fg_j = flowdir.create_flowgraph(dirs, dir_scheme=dir_scheme)
    np.testing.assert_array_equal(fg_i, expected_fg_i)
    np.testing.assert_array_equal(fg_j, expected_fg_j)

    # Config 2
    dirs = np.array([[5, 1, 1], [5, 1, 1], [5, 1, 1]])
    expected_fg_i = np.array(
        [
            *(0, 0, np.nan),
            *(1, 1, np.nan),
            *(2, 2, np.nan),
        ]
    )
    expected_fg_j = np.array(
        [
            *(1, 2, np.nan),
            *(1, 2, np.nan),
            *(1, 2, np.nan),
        ]
    )
    with pytest.warns(UserWarning):
        fg_i, fg_j = flowdir.create_flowgraph(dirs, dir_scheme=dir_scheme)
    np.testing.assert_array_equal(fg_i, expected_fg_i)
    np.testing.assert_array_equal(fg_j, expected_fg_j)


def test_indegree_3x3():
    dir_scheme = D8Directions(transform_codes=lambda x: x)

    # Config 1
    dirs = np.array([[3, 3, 3], [3, 3, 3], [1, 1, 0]])

    expected_indegs = np.array(
        [
            [0, 0, 0],
            [1, 1, 1],
            [1, 2, 2],
        ]
    )

    np.testing.assert_array_equal(
        flowdir.count_indegree(dirs, dir_scheme=dir_scheme, backend="fortran"),
        expected_indegs,
    )

    # Config 2
    dirs = np.array([[5, 1, 1], [5, 1, 1], [5, 1, 1]])

    expected_indegs = np.array(
        [
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
        ]
    )

    np.testing.assert_array_equal(
        flowdir.count_indegree(dirs, dir_scheme=dir_scheme, backend="fortran"),
        expected_indegs,
    )


def test_strahler_order_3x3():
    dir_scheme = D8Directions(transform_codes=lambda x: x)

    # Config 1
    dirs = np.array(
        [
            [3, 3, 3],
            [3, 3, 3],
            [1, 1, 0],
        ]
    )

    expected_order = np.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 2, 2],
        ]
    )
    order = flowdir.compute_flow_strahler_order(dirs, dir_scheme=dir_scheme)

    np.testing.assert_array_equal(order, expected_order)

    # Config 2
    dirs = np.array(
        [
            [5, 1, 1],
            [5, 1, 1],
            [5, 1, 1],
        ]
    )

    expected_order = np.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]
    )

    order = flowdir.compute_flow_strahler_order(dirs, dir_scheme=dir_scheme)

    np.testing.assert_array_equal(order, expected_order)


def test_strahler_order_4x4():
    dir_scheme = D8Directions(transform_codes=lambda x: x)

    # Config 1
    dirs = np.array(
        [
            [1, 2, 2, 2],
            [8, 1, 1, 1],
            [8, 8, 8, 8],
            [1, 2, 1, 2],
        ]
    )
    expected_order = np.array(
        [
            [1, 2, 1, 1],
            [1, 1, 2, 2],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ]
    )
    order = flowdir.compute_flow_strahler_order(dirs, dir_scheme=dir_scheme)

    np.testing.assert_array_equal(order, expected_order)


def test_network_graph_3x3():
    dir_scheme = D8Directions(transform_codes=lambda x: x)

    dirs = np.array([[3, 3, 3], [3, 3, 3], [1, 1, 0]])
    valids = np.array([[T, F, T], [T, T, T], [T, T, T]])

    exp_orders = np.array([1, 1, 1, 2])
    exp_lengths = np.array([1, 2, 3, 1])
    exp_ijs = [
        np.array([[1, 1], [2, 1]]),
        np.array([[0, 2], [1, 2], [2, 2]]),
        np.array([[0, 0], [1, 0], [2, 0], [2, 1]]),
        np.array([[2, 1], [2, 2]]),
    ]
    arc_orders, vertex_ijs, arc_endpts = flowdir.construct_flowgraph(
        dirs, dir_scheme=dir_scheme, backend="fortran", min_order=1, valids=valids
    )
    arc_lengths = arc_endpts[:, 1] - arc_endpts[:, 0]

    np.testing.assert_array_equal(arc_orders, exp_orders)
    np.testing.assert_array_equal(arc_lengths, exp_lengths)

    for i, exp_ij in enumerate(exp_ijs):
        np.testing.assert_array_equal(
            vertex_ijs[arc_endpts[i, 0] : arc_endpts[i, 1] + 1], exp_ij
        )


def test_construct_flowgraph_fortran_returns_buffer_overflow_code():
    dirs = np.array([[1, 1, 0]], dtype=np.uint8, order="F")
    valids = np.ones((1, 3), dtype=bool, order="F")
    orders = np.ones((1, 3), dtype=np.int16, order="F")
    seeds = np.array([[True, False, False]], dtype=bool, order="F")
    indegs = np.array([[0, 1, 1]], dtype=np.int8, order="F")
    offsets = np.array([[0, 1], [0, 0]], dtype=np.int32, order="F")
    codes = np.array([1, 0], dtype=np.uint8, order="F")

    *_, err_code = graphs_f.construct_flowgraph(
        dirs,
        valids,
        orders,
        seeds,
        indegs,
        offsets,
        codes,
        True,
        1,
    )

    assert err_code == 3


def test_construct_flowgraph_translates_fortran_error(monkeypatch):
    def fake_construct(*args):
        return (
            0,
            0,
            np.zeros(1, dtype=np.int16),
            np.zeros((2, 2), dtype=np.int32),
            np.zeros((2, 1), dtype=np.int32),
            3,
        )

    monkeypatch.setattr(
        graphs_module,
        "graphs_f",
        SimpleNamespace(construct_flowgraph=fake_construct),
    )

    with pytest.raises(RuntimeError, match=r"construct_flowgraph.*error code 3"):
        flowdir.construct_flowgraph(
            np.array([[0]], dtype=np.uint8),
            orders=np.ones((1, 1), dtype=np.uint8),
            min_order=1,
            backend="fortran",
        )


@pytest.mark.parametrize(
    ("err_code", "exception", "detail"),
    [
        (1, ValueError, "invalid input"),
        (2, MemoryError, "allocate backend workspace"),
        (3, RuntimeError, "array or index capacity exceeded"),
        (99, RuntimeError, "unknown"),
    ],
)
def test_label_flats_translates_fortran_errors(
    monkeypatch, err_code, exception, detail
):
    def fake_label(*args):
        return np.zeros((1, 1), dtype=np.int32), err_code

    monkeypatch.setattr(
        raster_module,
        "raster_f",
        SimpleNamespace(label_flats=fake_label),
    )

    with pytest.raises(exception, match=rf"label_flats.*{detail}.*{err_code}"):
        raster_module.label_flats(
            np.zeros((1, 1), dtype=np.float32),
            np.ones((1, 1), dtype=bool),
        )


def test_max_branch_distance_translates_allocation_failure(monkeypatch):
    def fake_compute(*args):
        return np.zeros((1, 1), dtype=np.float32), 2

    monkeypatch.setattr(
        raster_module,
        "raster_f",
        SimpleNamespace(compute_max_branch_dist=fake_compute),
    )

    with pytest.raises(MemoryError, match=r"compute_max_branch_dist.*error code 2"):
        raster_module.compute_dist2conf_max(
            np.zeros((1, 1), dtype=np.uint8),
        )


def test_locate_invalid_graph_topogtaphy():
    vs = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
    endpts = np.array([[0, 1], [2, 3]])
    exp_intxs = np.array([[0, 1, 0, 2, 1]], dtype=np.int32)
    intxs = flowdir.locate_invalid_graph_topology(vs, endpts, backend="fortran")
    np.testing.assert_array_equal(intxs, exp_intxs)

    vs = np.array([[0, 0], [1, 1], [2, 0], [1, 0], [0, 1]])
    endpts = np.array([[0, 2], [3, 4]])
    exp_intxs = np.array([[0, 1, 0, 3, 1]], dtype=np.int32)
    intxs = flowdir.locate_invalid_graph_topology(vs, endpts, backend="fortran")
    np.testing.assert_array_equal(intxs, exp_intxs)

    # Test self-intersection within a single arc
    vs = np.array([[0, 0], [2, 2], [2, 0], [0, 2]])
    endpts = np.array([[0, 3]])
    exp_intxs = np.array([[0, 0, 0, 2, 1]], dtype=np.int32)
    intxs = flowdir.locate_invalid_graph_topology(vs, endpts, backend="fortran")
    np.testing.assert_array_equal(intxs, exp_intxs)

    # Test no violations
    vs = np.array([[0, 0], [1, 1], [2, 2]])
    endpts = np.array([[0, 2]])
    assert flowdir.locate_invalid_graph_topology(vs, endpts, backend="fortran") is None

    # Test error handling on invalid shapes
    with pytest.raises(ValueError, match="Invalid array shapes passed"):
        flowdir.locate_invalid_graph_topology(
            np.array([1, 2, 3]), endpts, backend="fortran"
        )


def _make_separated_x_pairs(
    npairs: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct isolated two-segment X crossings for locator regression tests.

    Each pair contributes exactly one intersection, and the spacing between
    pairs prevents unintended intersections.
    """
    vertices = []
    endpts = []
    for ipair in range(npairs):
        x = 3 * ipair
        start = len(vertices)
        vertices.extend(
            [
                [x, 0],
                [x + 1, 1],
                [x, 1],
                [x + 1, 0],
            ]
        )
        endpts.extend([[start, start + 1], [start + 2, start + 3]])

    return (
        np.asarray(vertices, dtype=np.float32),
        np.asarray(endpts, dtype=np.int32),
    )


def _scan_topology_with_capacity(
    vertices: np.ndarray,
    endpts: np.ndarray,
    capacity: int,
) -> tuple[np.ndarray, int, int]:
    """
    Call the low-level scanner after converting arrays to its FORTRAN layout.
    """
    vertices_f = np.asfortranarray(vertices.T, dtype=np.float32)
    endpts_f = np.asfortranarray(endpts.T + 1, dtype=np.int32)
    return graphs_f.scan_invalid_graph_topology(vertices_f, endpts_f, capacity)


def test_locate_invalid_graph_topology_retries_after_buffer_overflow():
    """
    The public FORTRAN backend returns all results after provisional overflow.
    """
    vertices, endpts = _make_separated_x_pairs()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        intxs_f = flowdir.locate_invalid_graph_topology(
            vertices, endpts, backend="fortran"
        )

    intxs_py = flowdir.locate_invalid_graph_topology(vertices, endpts, backend="python")
    assert not caught
    assert intxs_f is not None
    assert intxs_f.shape == (5, 5)
    assert intxs_f.dtype == np.int32
    assert intxs_f.flags.c_contiguous
    np.testing.assert_array_equal(intxs_f, intxs_py)


def test_topology_scanner_counts_past_capacity():
    """
    The low-level scanner reports the exact count beyond storage capacity.
    """
    vertices, endpts = _make_separated_x_pairs()
    intxs, nintxs, err_code = _scan_topology_with_capacity(vertices, endpts, capacity=1)

    assert err_code == 0
    assert nintxs == 5
    assert intxs.shape == (5, 1)
    np.testing.assert_array_equal(intxs[:, 0], [1, 2, 1, 3, 1])


@pytest.mark.parametrize("capacity", [4, 5, 8])
def test_topology_scanner_capacity_boundaries(capacity):
    """
    Stored records and counts are correct around the exact capacity.
    """
    vertices, endpts = _make_separated_x_pairs()
    intxs, nintxs, err_code = _scan_topology_with_capacity(vertices, endpts, capacity)

    assert err_code == 0
    assert nintxs == 5
    assert intxs.shape == (5, capacity)

    nstored = min(nintxs, capacity)
    public_intxs = flowdir.locate_invalid_graph_topology(
        vertices, endpts, backend="fortran"
    )
    assert public_intxs is not None
    expected_stored = public_intxs[:nstored].T.copy()
    expected_stored[:-1] += 1
    np.testing.assert_array_equal(intxs[:, :nstored], expected_stored)


def test_topology_scanner_empty_input_initialises_outputs():
    """
    An empty graph produces initialized scanner outputs and public `None`.
    """
    vertices = np.empty((0, 2), dtype=np.float32)
    endpts = np.empty((0, 2), dtype=np.int32)
    intxs, nintxs, err_code = _scan_topology_with_capacity(vertices, endpts, capacity=3)

    assert err_code == 0
    assert nintxs == 0
    assert intxs.shape == (5, 3)
    assert (
        flowdir.locate_invalid_graph_topology(vertices, endpts, backend="fortran")
        is None
    )


def test_topology_wrapper_uses_single_scan_when_capacity_is_sufficient(monkeypatch):
    """
    The wrapper avoids retrying when its provisional buffer is sufficient.
    """
    calls = []

    def fake_scan(vertex_ijs, arc_endpts, capacity):
        calls.append(capacity)
        intxs = np.empty((5, capacity), dtype=np.int32, order="F")
        intxs[:, 0] = [1, 2, 1, 3, 1]
        return intxs, 1, 0

    monkeypatch.setattr(
        graphs_module,
        "graphs_f",
        SimpleNamespace(scan_invalid_graph_topology=fake_scan),
    )
    intxs = flowdir.locate_invalid_graph_topology(
        np.zeros((4, 2)), np.array([[0, 1], [2, 3]]), backend="fortran"
    )

    assert calls == [3]
    np.testing.assert_array_equal(intxs, [[0, 1, 0, 2, 1]])


def test_topology_wrapper_retries_with_exact_reported_capacity(monkeypatch):
    """
    An overflow retry uses the total reported by the provisional scan.
    """
    calls = []

    def fake_scan(vertex_ijs, arc_endpts, capacity):
        calls.append(capacity)
        intxs = np.empty((5, capacity), dtype=np.int32, order="F")
        nintxs = 5
        nstored = min(capacity, nintxs)
        for i in range(nstored):
            intxs[:, i] = [2 * i + 1, 2 * i + 2, 2 * i + 1, 2 * i + 3, 1]
        return intxs, nintxs, 0

    monkeypatch.setattr(
        graphs_module,
        "graphs_f",
        SimpleNamespace(scan_invalid_graph_topology=fake_scan),
    )
    intxs = flowdir.locate_invalid_graph_topology(
        np.zeros((20, 2)),
        np.arange(20, dtype=np.int32).reshape(10, 2),
        backend="fortran",
    )

    assert calls == [3, 5]
    assert intxs is not None
    assert intxs.shape == (5, 5)


def test_topology_wrapper_rejects_inconsistent_retry_count(monkeypatch):
    """
    A changed count during the deterministic retry raises an error.
    """
    calls = []

    def fake_scan(vertex_ijs, arc_endpts, capacity):
        calls.append(capacity)
        intxs = np.empty((5, capacity), dtype=np.int32, order="F")
        return intxs, 4 if len(calls) == 1 else 3, 0

    monkeypatch.setattr(
        graphs_module,
        "graphs_f",
        SimpleNamespace(scan_invalid_graph_topology=fake_scan),
    )
    with pytest.raises(RuntimeError, match="count changed"):
        flowdir.locate_invalid_graph_topology(
            np.zeros((4, 2)),
            np.array([[0, 1], [2, 3]]),
            backend="fortran",
        )

    assert calls == [3, 4]


@pytest.mark.parametrize(
    ("err_code", "exception"),
    [(1, ValueError), (2, MemoryError), (99, RuntimeError)],
)
def test_topology_wrapper_translates_scanner_errors(monkeypatch, err_code, exception):
    """
    Scanner status codes map to the documented Python exceptions.
    """

    def fake_scan(vertex_ijs, arc_endpts, capacity):
        return np.empty((5, capacity), dtype=np.int32), 0, err_code

    monkeypatch.setattr(
        graphs_module,
        "graphs_f",
        SimpleNamespace(scan_invalid_graph_topology=fake_scan),
    )
    with pytest.raises(exception):
        flowdir.locate_invalid_graph_topology(
            np.zeros((2, 2)), np.array([[0, 1]]), backend="fortran"
        )


def test_self_intersection_overflow_does_not_hide_later_interarc_results():
    """
    Self-scan overflow does not hide later inter-arc violations.
    """
    vertices = np.array(
        [
            [0, 0],
            [1, 1],
            [0, 0],
            [1, 1],
            [0, 0],
            [1, 1],
            [0, 0],
            [10, 0],
            [11, 1],
            [10, 1],
            [11, 0],
        ],
        dtype=np.float32,
    )
    endpts = np.array([[0, 6], [7, 8], [9, 10]], dtype=np.int32)

    intxs_f = flowdir.locate_invalid_graph_topology(vertices, endpts, backend="fortran")
    intxs_py = flowdir.locate_invalid_graph_topology(vertices, endpts, backend="python")

    assert intxs_f is not None
    assert intxs_f.shape[0] > 3
    assert np.any((intxs_f[:, 0] == 0) & (intxs_f[:, 1] == 0))
    assert np.any((intxs_f[:, 0] == 1) & (intxs_f[:, 1] == 2))
    np.testing.assert_array_equal(intxs_f, intxs_py)


def test_large_self_intersection_only_result_matches_python_backend():
    """
    A dense self-intersection result stays complete across backends.
    """
    vertices = np.array(
        [[0, 0], [1, 1], [0, 0], [1, 1], [0, 0], [1, 1], [0, 0]],
        dtype=np.float32,
    )
    endpts = np.array([[0, 6]], dtype=np.int32)

    intxs_f = flowdir.locate_invalid_graph_topology(vertices, endpts, backend="fortran")
    intxs_py = flowdir.locate_invalid_graph_topology(vertices, endpts, backend="python")

    assert intxs_f is not None
    assert intxs_f.shape[0] > 3
    assert np.all(intxs_f[:, :2] == 0)
    np.testing.assert_array_equal(intxs_f, intxs_py)


def test_topology_results_are_invariant_to_arc_reordering():
    """
    Arc reordering preserves the geometric intersection set.
    """
    vertices, endpts = _make_separated_x_pairs()
    baseline = flowdir.locate_invalid_graph_topology(
        vertices, endpts, backend="fortran"
    )

    permutation = np.arange(endpts.shape[0] - 1, -1, -1)
    reordered = flowdir.locate_invalid_graph_topology(
        vertices, endpts[permutation], backend="fortran"
    )

    def remap_rows(rows, arc_ids):
        remapped = set()
        for iarc, jarc, iseg, jseg, intx_flag in rows:
            iarc = int(arc_ids[iarc])
            jarc = int(arc_ids[jarc])
            if iarc > jarc:
                iarc, jarc = jarc, iarc
                iseg, jseg = jseg, iseg
            remapped.add((iarc, jarc, int(iseg), int(jseg), int(intx_flag)))
        return remapped

    assert remap_rows(baseline, np.arange(endpts.shape[0])) == remap_rows(
        reordered, permutation
    )


def test_topology_repair_simplifies_each_conflicting_arc_once(monkeypatch):
    """
    Each conflicting arc is simplified once per repair iteration.
    """
    intersections = np.array(
        [
            [0, 1, 0, 2, 1],
            [0, 1, 0, 2, 4],
            [0, 2, 0, 4, 1],
            [1, 2, 2, 4, 1],
        ],
        dtype=np.int32,
    )
    locator_results = iter([intersections, None])
    simplified_starts = []

    def fake_locator(vertex_xys, arc_endpts, backend="fortran"):
        return next(locator_results)

    def fake_simplify(vertex_xys, arc_endpts, tol):
        simplified_starts.append(tuple(vertex_xys[:, 0]))
        return np.ones(vertex_xys.shape[1], dtype=np.int8)

    monkeypatch.setattr(graphs_module, "locate_invalid_graph_topology", fake_locator)
    monkeypatch.setattr(
        graphs_module,
        "graphs_f",
        SimpleNamespace(simplify_flowgraph=fake_simplify),
    )

    vertices = np.array([[0, 1, 2, 3, 4, 5], [0, 0, 0, 0, 0, 0]], dtype=np.float32)
    endpts = np.array([[0, 2, 4], [1, 3, 5]], dtype=np.int32)
    keeps = graphs_module._resolve_topology_intersections(
        vertices, endpts, np.ones(6, dtype=bool), tol=1.0, max_iters=1
    )

    assert simplified_starts == [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)]
    np.testing.assert_array_equal(keeps, np.ones(6, dtype=bool))


def test_topology_repair_attempt_count_matches_max_iters(monkeypatch):
    intersections = np.array([[0, 1, 0, 2, 1]], dtype=np.int32)
    simplify_calls = []

    monkeypatch.setattr(
        graphs_module,
        "locate_invalid_graph_topology",
        lambda *args, **kwargs: intersections,
    )

    def fake_simplify(vertex_xys, arc_endpts, tol):
        simplify_calls.append(tol)
        return np.ones(vertex_xys.shape[1], dtype=np.int8)

    monkeypatch.setattr(
        graphs_module,
        "graphs_f",
        SimpleNamespace(simplify_flowgraph=fake_simplify),
    )

    vertices = np.array([[0, 1, 2, 3], [0, 0, 0, 0]], dtype=np.float32)
    endpts = np.array([[0, 2], [1, 3]], dtype=np.int32)
    graphs_module._resolve_topology_intersections(
        vertices,
        endpts,
        np.ones(4, dtype=bool),
        tol=1.0,
        max_iters=0,
    )

    assert simplify_calls == []


def test_simplify_single_flowgraph():
    verts = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    endpts = np.array([[0, 2]])
    orders = np.array([2, 4])
    simp_orders, simp_verts, simp_endpts, keeps = flowdir.simplify_flowgraph(
        np.array([3]), verts, endpts, tol=1.0, check_topology=False, backend="fortran"
    )
    np.testing.assert_array_equal(simp_orders, [3])
    np.testing.assert_array_equal(keeps, [True, False, True])
    np.testing.assert_array_equal(simp_verts, [[0.0, 0.0], [2.0, 2.0]])
    np.testing.assert_array_equal(simp_endpts, [[0, 1]])

    # Test topology correction
    verts_topo = np.array([[0.0, 0.8], [1.0, 2.0], [2.0, 0.2], [0.5, 0.5], [1.5, 0.5]])
    endpts_topo = np.array([[0, 2], [3, 4]])

    # Under tol = 1.5 and check_topology = False, simplification occurs and causes intersection
    _, _, _, keeps_no_check = flowdir.simplify_flowgraph(
        *(orders, verts_topo, endpts_topo),
        tol=1.5,
        check_topology=False,
        backend="fortran",
    )
    # Vertex 1 should be removed
    np.testing.assert_array_equal(keeps_no_check, [True, False, True, True, True])

    # Under tol = 1.5 and check_topology = True, it detects intersection, reduces tolerance,
    # and keeps Vertex 1 to avoid intersection
    _, checked_verts, checked_endpts, keeps_with_check = flowdir.simplify_flowgraph(
        *(orders, verts_topo, endpts_topo),
        tol=1.5,
        check_topology=True,
        backend="fortran",
    )
    # Vertex 1 should be kept
    np.testing.assert_array_equal(keeps_with_check, [True, True, True, True, True])
    assert (
        flowdir.locate_invalid_graph_topology(
            checked_verts, checked_endpts, backend="fortran"
        )
        is None
    )

    with warnings.catch_warnings():
        warnings.simplefilter("default")
        filters_before = list(warnings.filters)
        flowdir.simplify_flowgraph(
            *(orders, verts_topo, endpts_topo),
            tol=1.5,
            check_topology=True,
            backend="fortran",
        )
        assert warnings.filters == filters_before


def test_simplify_single_flowgraph_preserves_vertex_layout():
    verts = np.array(
        [
            [0.0, 1.0],
            [1.0, 3.0],
            [2.0, 1.0],
            [3.0, 4.0],
        ]
    )
    endpts = np.array([[0, 3]])
    orders = np.array([5])

    simp_orders, simp_verts, simp_endpts, keeps = flowdir.simplify_flowgraph(
        *(orders, verts, endpts),
        tol=0.0,
        check_topology=False,
        backend="fortran",
    )

    assert simp_verts.shape == (4, 2)
    np.testing.assert_array_equal(simp_orders, orders)
    np.testing.assert_array_equal(simp_verts, verts)
    np.testing.assert_array_equal(simp_endpts, endpts)
    np.testing.assert_array_equal(keeps, np.ones(4, dtype=bool))


def test_simplify_flowgraph_validates_arc_orders():
    verts = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    endpts = np.array([[0, 2]])
    orders = [np.array([1]), np.array([2])]

    with pytest.raises(TypeError, match="must be NumPy arrays"):
        flowdir.simplify_flowgraph(
            np.array([1]),
            "not-an-array",  # type: ignore
            endpts,
            check_topology=False,
        )

    with pytest.raises(ValueError, match="Order array has length 0"):
        flowdir.simplify_flowgraph(
            np.array([], dtype=np.uint8),
            verts,
            endpts,
            check_topology=False,
        )

    with pytest.raises(ValueError, match="must have the same length"):
        flowdir.simplify_flowgraph(
            *(orders, [verts], [endpts]),
            check_topology=False,
        )


def test_simplify_rejects_invalid_final_graph_from_valid_input(monkeypatch):
    verts = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [0.5, 0.5], [1.5, 0.5]])
    endpts = np.array([[0, 2], [3, 4]])
    orders = np.array([1, 2])
    final_intxs = np.array([[0, 1, 0, 2, 1]], dtype=np.int32)

    monkeypatch.setattr(
        graphs_module,
        "_resolve_topology_intersections",
        lambda verts, endpts, keeps, tol, graph_ids=None: np.array([T, F, T, T, T]),
    )
    locator_results = iter([final_intxs, None])
    monkeypatch.setattr(
        graphs_module,
        "_locate_disallowed_graph_topology",
        lambda verts, endpts, graph_ids=None: next(locator_results),
    )

    with pytest.raises(flowdir.UnresolvedSimplificationTopology) as exc_info:
        flowdir.simplify_flowgraph(
            *(orders, verts, endpts),
            tol=1.0,
            check_topology=True,
            backend="fortran",
        )


def test_simplify_rejects_invalid_final_graph_from_invalid_input(monkeypatch):
    verts = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    endpts = np.array([[0, 1], [2, 3]])
    orders = np.array([1, 2])
    final_intxs = np.array([[0, 1, 0, 2, 1]], dtype=np.int32)
    input_intxs = np.array([[0, 1, 0, 2, 1]], dtype=np.int32)

    monkeypatch.setattr(
        graphs_module,
        "_resolve_topology_intersections",
        lambda verts, endpts, keeps, tol, graph_ids=None: np.ones(
            verts.shape[1], dtype=bool
        ),
    )
    locator_results = iter([final_intxs, input_intxs])
    monkeypatch.setattr(
        graphs_module,
        "_locate_disallowed_graph_topology",
        lambda verts, endpts, graph_ids=None: next(locator_results),
    )

    with pytest.raises(flowdir.InvalidOriginalGraphTopology) as exc_info:
        flowdir.simplify_flowgraph(
            *(orders, verts, endpts),
            tol=1.0,
            check_topology=True,
            backend="fortran",
        )


def test_simplify_skips_final_validation_when_topology_check_is_disabled(
    monkeypatch,
):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("Topology validation should be disabled.")

    monkeypatch.setattr(
        graphs_module, "_locate_disallowed_graph_topology", fail_if_called
    )

    flowdir.simplify_flowgraph(
        np.array([1]),
        np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
        np.array([[0, 2]]),
        tol=1.0,
        check_topology=False,
        backend="fortran",
    )


def test_simplify_multiple_flowgraphs():
    # Llist of standard arrays
    vs0 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    endpts0 = np.array([[0, 2]], dtype=np.int32)
    orders0 = np.array([2], dtype=np.uint8)
    verts1 = np.array([[3.0, 3.0], [4.0, 4.0], [5.0, 5.0]], dtype=np.float32)
    endpts1 = np.array([[0, 2]], dtype=np.int32)
    orders1 = np.array([4], dtype=np.uint8)

    simp_orders_list, simp_verts_list, simp_endpts_list, keeps_list = (
        flowdir.simplify_flowgraph(
            [orders0, orders1],
            [vs0, verts1],
            [endpts0, endpts1],
            tol=1.0,
            check_topology=False,
            backend="fortran",
        )
    )
    assert isinstance(simp_orders_list, list)
    assert isinstance(simp_verts_list, list)
    np.testing.assert_array_equal(simp_orders_list[0], orders0)
    np.testing.assert_array_equal(simp_orders_list[1], orders1)
    np.testing.assert_array_equal(keeps_list[0], [T, F, T])
    np.testing.assert_array_equal(keeps_list[1], [T, F, T])
    np.testing.assert_array_equal(simp_verts_list[0], [[0.0, 0.0], [2.0, 2.0]])
    np.testing.assert_array_equal(simp_verts_list[1], [[3.0, 3.0], [5.0, 5.0]])
    np.testing.assert_array_equal(simp_endpts_list[0], [[0, 1]])
    np.testing.assert_array_equal(simp_endpts_list[1], [[0, 1]])

    # Tuple of transposed/differing shapes
    verts0_t = vs0.T  # shape (2, 3)
    endpts0_t = endpts0.T  # shape (2, 1)

    simp_orders_tuple, simp_vs_tuple, simp_endpts_tuple, keeps_tuple = (
        flowdir.simplify_flowgraph(
            *((orders0, orders1), (verts0_t, verts1), (endpts0_t, endpts1)),
            tol=1.0,
            check_topology=False,
            backend="fortran",
        )
    )
    assert isinstance(simp_orders_tuple, tuple)
    assert isinstance(simp_vs_tuple, tuple)
    np.testing.assert_array_equal(keeps_tuple[0], [T, F, T])
    np.testing.assert_array_equal(keeps_tuple[1], [T, F, T])
    # Verify orientation restoration
    assert simp_vs_tuple[0].shape == (2, 2)
    assert simp_endpts_tuple[0].shape == (2, 1)
    assert simp_vs_tuple[1].shape == (2, 2)
    assert simp_endpts_tuple[1].shape == (1, 2)


def test_simplify_multiple_flowgraphs_accepts_one_empty_graph():
    empty_orders = np.empty((0,), dtype=np.uint8)
    empty_verts = np.empty((0, 2), dtype=np.float32)
    empty_endpts = np.empty((0, 2), dtype=np.int32)
    orders = np.array([1], dtype=np.uint8)
    verts = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    endpts = np.array([[0, 1]], dtype=np.int32)

    simp_orders, simp_verts, simp_endpts, keeps = flowdir.simplify_flowgraph(
        [empty_orders, orders],
        [empty_verts, verts],
        [empty_endpts, endpts],
        tol=0.0,
        check_topology=True,
        backend="fortran",
    )

    np.testing.assert_array_equal(simp_orders[0], empty_orders)
    np.testing.assert_array_equal(simp_verts[0], empty_verts)
    np.testing.assert_array_equal(simp_endpts[0], empty_endpts)
    np.testing.assert_array_equal(keeps[0], np.empty((0,), dtype=bool))
    np.testing.assert_array_equal(simp_orders[1], orders)
    np.testing.assert_array_equal(simp_verts[1], verts)
    np.testing.assert_array_equal(simp_endpts[1], endpts)
    np.testing.assert_array_equal(keeps[1], np.ones(2, dtype=bool))


@pytest.mark.parametrize("collection_type", (list, tuple))
def test_simplify_multiple_flowgraphs_round_trips_all_empty_graphs(collection_type):
    orders = collection_type(
        (np.empty((0,), dtype=np.uint8), np.empty((0,), dtype=np.int16))
    )
    verts = collection_type(
        (np.empty((0, 2), dtype=np.float32), np.empty((2, 0), dtype=np.float64))
    )
    endpts = collection_type(
        (np.empty((0, 2), dtype=np.int32), np.empty((2, 0), dtype=np.int64))
    )

    simp_orders, simp_verts, simp_endpts, keeps = flowdir.simplify_flowgraph(
        orders,
        verts,
        endpts,
        tol=1.0,
        check_topology=True,
        backend="fortran",
    )

    assert isinstance(simp_orders, collection_type)
    assert isinstance(simp_verts, collection_type)
    assert isinstance(simp_endpts, collection_type)
    assert isinstance(keeps, collection_type)
    for original_group, simplified_group in zip(
        (orders, verts, endpts),
        (simp_orders, simp_verts, simp_endpts),
    ):
        for original, simplified in zip(original_group, simplified_group):
            np.testing.assert_array_equal(simplified, original)
            assert simplified.shape == original.shape
            assert simplified.dtype == original.dtype
    for keep in keeps:
        np.testing.assert_array_equal(keep, np.empty((0,), dtype=bool))


@pytest.mark.parametrize("collection_type", (list, tuple))
def test_simplify_multiple_flowgraphs_accepts_empty_collections(collection_type):
    empty = collection_type()

    results = flowdir.simplify_flowgraph(empty, empty, empty)

    for result in results:
        assert isinstance(result, collection_type)
        assert len(result) == 0


def test_simplify_flowgraph_keeps_every_arc_endpoint():
    orders = np.array([1, 2, 3], dtype=np.uint8)
    verts = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.2],
            [2.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.2],
            [4.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
        ]
    )
    endpts = np.array([[0, 2], [3, 5], [6, 7]], dtype=np.int32)

    _, simp_verts, simp_endpts, keeps = flowdir.simplify_flowgraph(
        orders,
        verts,
        endpts,
        tol=1.0,
        check_topology=True,
        backend="fortran",
    )

    assert np.all(keeps[endpts.ravel()])
    np.testing.assert_array_equal(
        simp_verts[simp_endpts.ravel()],
        verts[endpts.ravel()],
    )


def test_simplify_multiple_flowgraphs_inserts_overlap_endpoints():
    vertices = [
        np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        np.array([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]]),
    ]
    endpts = [np.array([[0, 2]]), np.array([[0, 2]])]
    orders = [np.array([3]), np.array([5])]

    simp_orders, simp_verts, simp_endpts, keeps = flowdir.simplify_flowgraph(
        *(orders, vertices, endpts),
        tol=0.0,
        check_topology=True,
        backend="fortran",
    )
    np.testing.assert_array_equal(simp_orders[0], [3, 3])
    np.testing.assert_array_equal(simp_orders[1], [5, 5])

    expected_start_verts = [
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        np.array([[1.0, -1.0], [1.0, 0.0]]),
    ]
    for graph_verts, graph_endpts, graph_keeps, expected_starts in zip(
        simp_verts, simp_endpts, keeps, expected_start_verts
    ):
        assert graph_endpts.shape == (2, 2)
        np.testing.assert_array_equal(graph_endpts, np.array([[0, 1], [2, 3]]))
        np.testing.assert_array_equal(graph_verts[graph_endpts[:, 0]], expected_starts)
        # Batch overlap splitting rebuilds a compact vertex array, duplicating
        # only the shared endpoint required by the two resulting arcs.
        assert graph_keeps.size == 4

    # Both occurrences of the crossing are endpoints after each graph is split
    for graph_verts, graph_endpts in zip(simp_verts, simp_endpts):
        endpoint_verts = graph_verts[graph_endpts.ravel()]
        assert np.sum(np.all(endpoint_verts == [1.0, 0.0], axis=1)) == 2


def test_simplify_multiple_flowgraphs_ignores_identical_arcs():
    verts = [
        np.array([[0.0, 0.0], [1.0, 0.2], [2.0, 0.4], [3.0, 0.2], [4.0, 0.0]]),
        np.array([[4.0, 0.0], [3.0, 0.2], [2.0, 0.4], [1.0, 0.2], [0.0, 0.0]]),
    ]
    endpts = [np.array([[0, 4]]), np.array([[0, 4]])]
    orders = [np.array([2]), np.array([4])]

    _, simp_verts, _, _ = flowdir.simplify_flowgraph(
        *(orders, verts, endpts),
        tol=0.25,
        check_topology=True,
        backend="fortran",
    )

    # The identical central arcs may simplify despite having opposite directions
    for graph_verts in simp_verts:
        assert not np.any(np.all(graph_verts == [2.0, 0.4], axis=1))


@pytest.mark.parametrize(
    ("err_code", "exception"),
    [
        (1, ValueError),
        (2, MemoryError),
        (3, RuntimeError),
        (99, RuntimeError),
    ],
)
def test_find_acyclic_flowdirs_translates_fortran_errors(
    monkeypatch, err_code, exception
):
    def fake_find(*args):
        return np.zeros((1, 1), dtype=bool), err_code

    monkeypatch.setattr(
        raster_module,
        "raster_f",
        SimpleNamespace(find_acyclic_flowdirs=fake_find),
    )

    with pytest.raises(exception):
        flowdir.find_acyclic_flowdirs(
            np.zeros((1, 1), dtype=np.uint8),
            indegs=np.zeros((1, 1), dtype=np.int8),
            backend="fortran",
        )


if __name__ == "__main__":
    test_priority_queue_pushes_into_empty_queue_and_pops_in_elevation_order()
    test_priority_queue_matches_sorted_random_elevations()
    test_priority_queue_handles_equal_elevations_and_reports_invalid_operations()
    test_downstreamid_3x3()
    test_downstreamid_4x4()
    test_flowgraph_3x3()
    test_indegree_3x3()
    test_strahler_order_3x3()
    test_strahler_order_4x4()
    test_network_graph_3x3()
    test_locate_invalid_graph_topogtaphy()
    test_simplify_single_flowgraph()
    test_simplify_multiple_flowgraphs()
    test_locate_invalid_graph_topology_retries_after_buffer_overflow()
    test_topology_scanner_counts_past_capacity()
    test_simplify_multiple_flowgraphs_inserts_overlap_endpoints()
    test_simplify_multiple_flowgraphs_ignores_identical_arcs()
    test_simplify_flowgraph_keeps_every_arc_endpoint()
    test_simplify_multiple_flowgraphs_accepts_one_empty_graph()
    test_label_mask_areas()
