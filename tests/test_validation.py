"""
Tests array and geomorphological raster validation routines.

Created: 2026-08-23, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from formosa.geomorphology.raster_validation import (
    validate_format_dem,
    validate_format_flowdirs,
    validate_format_freeform_coordinates,
    validate_format_valids,
)
from formosa.utils import NpFlowDir, NpReal
from formosa.utils.validation import (
    validate_2d_raster,
    validate_array,
    validate_same_shape,
    validate_shape,
)


def test_validate_array_rejects_non_array_with_readable_message():
    with pytest.raises(TypeError, match="Values must be a NumPy array, but got type"):
        validate_array([1, 2], "values")  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("shape", [(0, 2), (2, 0), (2,), (1, 1, 1)])
def test_validate_2d_raster_rejects_empty_or_non_2d_arrays(shape: tuple[int, ...]):
    with pytest.raises(ValueError, match="non-empty 2D array"):
        validate_2d_raster(np.empty(shape), "raster")


def test_shape_validators_report_mismatches():
    first = np.zeros((2, 3))
    second = np.zeros((3, 2))

    with pytest.raises(ValueError, match=r"expected to be \(3, 2\)"):
        validate_shape(first, (3, 2), "first")
    with pytest.raises(ValueError, match="Shapes for first and second must match"):
        validate_same_shape(first, second, "first", "second")


@pytest.mark.parametrize("dem", [np.array([["land"]]), np.ones((1, 1), complex)])
def test_validate_format_dem_rejects_non_real_dtypes(dem: NDArray[NpReal]):
    with pytest.raises(TypeError):
        dem = validate_format_dem(dem)


def test_validate_format_valids_combines_mask_with_finite_cells_without_mutation():
    dem = np.array([[1.0, np.nan], [np.inf, 2.0]])
    supplied = np.array([[True, True], [True, False]])
    original = supplied.copy()

    result = validate_format_valids(supplied, dem, "DEM")

    np.testing.assert_array_equal(result, [[True, False], [False, False]])
    np.testing.assert_array_equal(supplied, original)


def test_validate_format_valids_requires_a_mask_or_reference():
    with pytest.raises(ValueError, match="Cannot determine"):
        valids = validate_format_valids(None, None)  # pyright: ignore[reportUnusedVariable]


def test_validate_format_flowdirs_converts_representable_integers_to_uint8():
    result = validate_format_flowdirs(
        np.array([[0, 128, 255]], dtype=np.int16)  # pyright: ignore[reportArgumentType]
    )

    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result, [[0, 128, 255]])


@pytest.mark.parametrize("value", [-1, 256])
def test_validate_format_flowdirs_rejects_values_outside_uint8_range(
    value: NDArray[NpFlowDir],
):
    with pytest.raises(ValueError, match=r"range \[0, 255\]"):
        dirs = validate_format_flowdirs(
            np.array([[value]], dtype=np.int16)  # pyright: ignore[reportArgumentType]
        )


def test_validate_format_flowdirs_requires_2d_even_with_reference_array():
    with pytest.raises(ValueError, match="non-empty 2D array"):
        dirs = validate_format_flowdirs(
            np.array([0, 1], dtype=np.uint8),
            against=np.ones(2),
        )


def test_freeform_coordinates_generate_requested_shape_and_dtype():
    x, y = validate_format_freeform_coordinates(
        None, None, shape=(2, 3), dtype=np.float32
    )

    assert x.dtype == np.float32
    assert y.dtype == np.float32
    np.testing.assert_array_equal(x, [[0, 1, 2], [0, 1, 2]])
    np.testing.assert_array_equal(y, [[0, 0, 0], [1, 1, 1]])


@pytest.mark.parametrize("supplied_axis", ["x", "y"])
def test_freeform_coordinates_preserve_one_supplied_coordinate(supplied_axis):
    supplied = np.full((2, 3), 7.5)
    x = supplied if supplied_axis == "x" else None
    y = supplied if supplied_axis == "y" else None

    result_x, result_y = validate_format_freeform_coordinates(x, y, shape=(2, 3))

    assert (result_x if supplied_axis == "x" else result_y) is supplied
    assert result_x.shape == result_y.shape == (2, 3)


@pytest.mark.parametrize("supplied_axis", ["x", "y"])
def test_freeform_coordinates_validate_one_supplied_coordinate_shape(supplied_axis):
    supplied = np.zeros((3, 2))
    x = supplied if supplied_axis == "x" else None
    y = supplied if supplied_axis == "y" else None

    with pytest.raises(ValueError, match=r"expected to be \(2, 3\)"):
        x, y = validate_format_freeform_coordinates(x, y, shape=(2, 3))


def test_freeform_coordinates_require_shape_when_an_axis_is_missing():
    with pytest.raises(ValueError, match="Cannot infer"):
        x, y = validate_format_freeform_coordinates(np.zeros((2, 3)), None)
