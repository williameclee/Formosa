# Style Guide

## Naming Conventions

All language should be in British English by default.

### Variables

Arrays should typically be plural nouns (e.g. `dirs`, `dists`), while other variable should be single nouns. This extends to boolean mask arrays, which should still be names as plural nouns (e.g. `valids` instead of `is_valid`).
Index arrays (containing pairs of ($i$, $j$) indices) should be named with `_ijs` suffix (e.g. `tofill_ijs`).

#### Standard Variable Names

Some variables are used so frequently that they should be named consistently across the codebase. These include:

- `dir_scheme`: instance of `D8Directions` defining the meaning of code numbers in flow direction arrays
- `dirs`: flow direction array
- `dists`: distance array
- `valids`: boolean mask array indicating valid cells
- `indegs`: indegree array for flow direction grid

### Functions

Function names should use snake_case, and is generally prefixed with a verb describing the action (e.g. `compute_`, `label_`).

## Documentation

### File Descriptions

Every substantive source file should begin with a short description
of the file's current purpose. A file is substantive when it
contains implementation code, rather than only re-exports or package
initialisation. Empty `__init__.py` files and `__init__.py` files
that only define the public import surface do not need a file
description.

The description should:

- State the file's primary responsibility in its first sentence;
- Describe important scope, assumptions, algorithms, or relationships to other
  modules only when these are not apparent from the first sentence;
- Identify an internal backend when the file is not intended to be used
  directly by users;
- Use British English, complete sentences, and the present tense; and
- Describe the file as it currently exists, rather than list its contents or
  repeat the documentation of individual functions and classes.
- Keep each line of the description to a maximum of 68 characters, and avoid
  trailing whitespace.

Keep the description brief: normally one summary sentence and no more than one
or two explanatory paragraphs. Longer algorithmic explanations belong in the
relevant function documentation, a reference document, or an architecture
decision record.

A `Last modified` line is encouraged. Git history is the authoritative record of
changes, so this line must not contain a modification log. When retained, use
the ISO 8601 date of the last substantive change to the file description or
implementation, followed by the contributor's preferred name. An email address
may be included but is not required:

```text
Last modified: YYYY-MM-DD, Name (email@domain)
```

Do not add per-change history fields. Record implementation details in commit messages and pull requests, user-visible changes in `CHANGELOG.md` or release notes, and consequential design decisions
in architecture decision records.

#### Python Files

Use the module docstring as the file description. Place it before imports and
follow the general conventions of PEP 257: a concise summary sentence, a blank
line, and any additional explanation. Use reStructuredText roles where a symbol
reference is useful because the documentation follows NumPy/Sphinx conventions.

```python
"""
Represents and process gridded digital elevation models.

This module provides :class:`DEMGrid`, which coordinates raster input and
geomorphological operations on a digital elevation model (DEM).

Last modified: 2026-08-09, En-Chi Lee (williameclee@gmail.com)
"""
```

For a private backend, make its role explicit:

```python
"""
Resolves flats in digital elevation models using the Python backend.

This module implements internal routines called by the public-facing drainage
API and is not intended to be used directly.
"""
```

#### FORTRAN Files

Place a documentation-comment block immediately before the `module` statement.
Use `!>` to begin the block and `!!` for its continuation lines so that tools
such as FORD and Doxygen can associate it with the module. Follow the same
content rules as for Python files, and state whether the module is an internal
backend called from Python or from other FORTRAN routines.

```fortran
!> Computes flow directions for digital elevation model rasters.
!!
!! This module provides the internal Fortran backend called by the Python
!! drainage API. It also contains raster-level analyses of the resulting flow
!! field; flow-graph operations are implemented in the network modules.
!!
!! Last modified: 2026-08-09, En-Chi Lee
module drainage_flowdir
    implicit none(type, external)
contains
```

Use `!>`/`!!` documentation comments for public modules and procedures, and
ordinary `!` comments for implementation notes that should not become part of
generated API documentation.

#### Test Files

Every substantive Python test file should begin with a module docstring that
states the behaviour or component under test and the test boundary. Use the
same formatting and optional `Last modified` field as other Python file
descriptions.

The summary should use an active, third-person verb such as `Tests`, `Verifies`,
or `Checks` and should distinguish among:

- tests of the public API or behaviour shared by all backends;
- tests specific to the Python backend;
- tests specific to the FORTRAN backend, including native error handling; and
- parity tests that run equivalent cases across both backends.

Do not begin with vague wording such as "Tests related to", enumerate every
test case, or describe implementation details that can change without altering
the module's scope. Add a short explanatory paragraph only when the filename
and summary do not adequately identify the boundary. Shared test helpers and
fixtures should instead describe what they provide.

Test filename suffixes should match their scope:

- `_f.py` for FORTRAN-backend tests;
- `_py.py` for Python-backend tests; and
- `_parity.py` for explicit cross-backend parity tests.

Files without one of these suffixes may test the public API across configured
backends or behaviour that is independent of a computational backend.

```python
"""
Tests flat resolution using the FORTRAN backend.

This module covers native results, boundary cases, and translation of FORTRAN
status codes by the public drainage API.

Last modified: 2026-08-10, En-Chi Lee (williameclee@gmail.com)
"""
```

```python
"""
Verifies flow-metric parity between the Python and FORTRAN backends.

Last modified: 2026-08-10, En-Chi Lee (williameclee@gmail.com)
"""
```

### Docstrings

All public functions should have descriptive docstrings written in the *NumPy style*. This includes:

- A clear, concise summary line.
- A **Parameters** section detailing variable names, types (e.g. `NDArray[int]`), and default values if optional.
  - Each parameter should start with the short phrase describing the parameter, without the leading 'A' or 'The', and end with a period.
  - More detailed descriptions of parameters can be provided in a separate paragraph after the short phrase.
  - For optional parameters, the default value should be specified at the end of the description as a new list item (e.g. '- Default backend is `"python"`').
- A **Returns** section specifying returned values, types, and descriptions of output structures.
  - The documentation should be in the same style as that for the Parameters section.
- A **Raises** section detailing exceptions raised and under what conditions.

### Type Annotations

Functions should be fully type-annotated:

- Standard type hints from typing (e.g. `Optional`, `Literal`) should be used when warranted (e.g. `tuple` if preferred over `Tuple` from the `typing` package).
- NumPy arrays should use `numpy.typing.NDArray` with the inner type specified (e.g. `npt.NDArray[np.integer]`, `npt.NDArray[np.number]`).

----
Last updated: Aug 10, 2026 ([@williameclee](https://github.com/williameclee))
