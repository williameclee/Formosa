"""
Defines common type aliases and TypeVars for array and scalar types.

This module provides reusable type definitions for NumPy indices,
coordinates, and numeric types used across the package.

Last modified: 2026-08-18, En-Chi Lee (williameclee@gmail.com)
"""

from typing import TypeVar, TypeAlias
import numpy as np

NpInt = TypeVar(
    "NpInt", np.uint8, np.int8, np.int16, np.uint16, np.int32, np.int64, np.intp
)
# Canonical type for NumPy array indices
NpCanonIndex: TypeAlias = np.int32
# Accpetable types for NumPy array indices
NpIndex = TypeVar("NpIndex", np.int32, np.int64, np.intp)
# Accpetable types for coordinates in NumPy arrays
NpCoords = TypeVar("NpCoords", np.int32, np.int64, np.intp, np.floating)
# Real number in NumPyArrays
NpReal = TypeVar("NpReal", np.integer, np.floating)
Real = int | float
