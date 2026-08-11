from typing import TypeVar, TypeAlias
import numpy as np

# Canonical type for NumPy array indices
NpCanonIndex: TypeAlias = np.int32
# Accpetable types for NumPy array indices
NpIndex = TypeVar("NpIndex", np.int32, np.int64, np.intp)
# Accpetable types for coordinates in NumPy arrays
NpCoords = TypeVar("NpCoords", np.int32, np.int64, np.intp, np.floating)
