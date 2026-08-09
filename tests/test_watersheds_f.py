# Last modified
#   2026-07-01, En-Chi Lee (williameclee@gmail.com)
#     - Added test cases for `create_flowgraph`, and
#       `compute_flow_strahler_order`.
#   2026-07-30, En-Chi Lee (williameclee@gmail.com)
#     - Various minor refactors and type annotation enhancements.
#   2026-08-04, En-Chi Lee (williameclee@gmail.com)
#     - Added test cases for FORTRAN error code handling.
#   2026-08-05, En-Chi Lee (williameclee@gmail.com)
#     - Added a regression check for nonstandard old-style FORTRAN
#       kind declarations.

from tests.core import *

import pytest
import re
from pathlib import Path
import numpy as np

from formosa import D8Directions
from formosa.geomorphology.flowdir import watersheds as wshed_m


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
    order = wshed_m.compute_flow_strahler_order(dirs, dir_scheme=dir_scheme)

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

    order = wshed_m.compute_flow_strahler_order(dirs, dir_scheme=dir_scheme)

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
    order = wshed_m.compute_flow_strahler_order(dirs, dir_scheme=dir_scheme)

    np.testing.assert_array_equal(order, expected_order)
