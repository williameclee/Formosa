"""
Defines common type aliases and TypeVars for array and scalar types.

This module provides reusable type definitions for NumPy indices,
coordinates, and numeric types used across the package.

Created: 2026-08-01, En-Chi Lee (williameclee@gmail.com)
Last modified: 2026-08-24, En-Chi Lee (williameclee@gmail.com)
"""

from typing import TypeAlias

import numpy as np

# Generic NumPy array dtype
type NpInt = (
    np.uint8
    | np.int8
    | np.int16
    | np.uint16
    | np.int32
    | np.int64
    | np.intp
)
# Canonical type for NumPy flow direction codes
NpFlowDir: TypeAlias = np.uint8
# Canonical type for NumPy array indices
NpCanonIndex: TypeAlias = np.int32
# Acceptable types for NumPy array indices
type NpIndex = np.int32 | np.int64 | np.intp
# Acceptable types for coordinates in NumPy arrays
type NpCoords = (
    np.int32 | np.int64 | np.intp | np.float32 | np.float64
)
Coords = int | float
# Real number in NumPy arrays
type NpReal = np.int8 | np.int16 | np.int32 | np.int64 | np.floating
Real = int | float
