"""
Validates array inputs and shapes for numeric operations.

This module implements internal validation routines for verifying
NumPy array types, 2D raster dimensions, and matching shapes.

Created: 2026-08-22, En-Chi Lee (williameclee@gmail.com)
Last modified: 2026-08-23, En-Chi Lee (williameclee@gmail.com)
"""

import numpy as np


def validate_array(arr: np.ndarray, arr_name: str = "array") -> None:
    """
    Validates that an input argument is a NumPy array.

    Parameters
    ----------
    arr : NDArray[number | bool]
        Input object to validate.
    arr_name : str, optional
        Name of the array for error messages.
        - Default name is `'array'`.
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(
            f"{arr_name.capitalize()} must be a NumPy array, "
            + f"but got type {type(arr)}.",
        )


def validate_2d_raster(arr: np.ndarray, arr_name: str = "array") -> None:
    """
    Validates that an array is a non-empty 2D raster.

    Parameters
    ----------
    arr : NDArray[number | bool]
        Array to validate.
        - Expected shape: `(nrows, ncols)`.
    arr_name : str, optional
        Name of the array for error messages.
        - Default name is `'array'`.
    """
    validate_array(arr, arr_name)
    if arr.ndim != 2 or 0 in arr.shape:
        raise ValueError(
            f"{arr_name.capitalize()} must be a non-empty 2D array, "
            + f"but received shape {arr.shape}."
        )


def validate_shape(
    arr: np.ndarray,
    shape: tuple[int, int],
    arr_name: str = "array",
) -> None:
    """
    Validates that an array matches an expected shape.

    Parameters
    ----------
    arr : NDArray[number | bool]
        Array to validate.
        - Expected shape: `shape`.
    shape : tuple[int, int]
        Expected shape tuple `(nrows, ncols)`.
    arr_name : str, optional
        Name of the array for error messages.
        - Default name is `'array'`.
    """
    validate_array(arr, arr_name)
    if arr.shape != shape:
        raise ValueError(
            f"Shape for {arr_name} is expected to be {shape}, "
            + f"but got shape {arr.shape}."
        )


def validate_same_shape(
    arr1: np.ndarray,
    arr2: np.ndarray,
    arr1_name: str = "array 1",
    arr2_name: str = "array 2",
) -> None:
    """
    Validates that two arrays have identical shapes.

    Parameters
    ----------
    arr1 : NDArray[number | bool]
        First array to validate.
        - Expected shape: `(nrows, ncols)`, same as `arr2`.
    arr2 : NDArray[number | bool]
        Second array to compare against.
        - Expected shape: `(nrows, ncols)`, same as `arr1`.
    arr1_name : str, optional
        Name of the first array for error messages.
        - Default name is `'array 1'`.
    arr2_name : str, optional
        Name of the second array for error messages.
        - Default name is `'array 2'`.
    """
    validate_array(arr1, arr1_name)
    validate_array(arr2, arr2_name)
    if arr1.shape != arr2.shape:
        raise ValueError(
            f"Shapes for {arr1_name} and {arr2_name} must match, "
            + f"but got shapes {arr1.shape} and {arr2.shape}, respectively."
        )


__all__ = [
    "validate_2d_raster",
    "validate_array",
    "validate_same_shape",
    "validate_shape",
]
