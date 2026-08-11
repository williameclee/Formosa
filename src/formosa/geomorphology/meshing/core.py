"""
Defines simple helpers shared within this module.

Created: 2026-08-11, En-Chi Lee (williameclee@gmail.com)
"""

from enum import IntFlag


class ConstraintKind(IntFlag):
    UNKNOWN = 0
    VALLEY = 1
    RIDGE = 2
    BOUNDARY = 4
