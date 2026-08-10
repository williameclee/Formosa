"""
Tests flow-based raster metrics using the FORTRAN backend.

Last modified: 2026-08-10, En-Chi Lee (williameclee@gmail.com)
"""

import re
from pathlib import Path


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
