# Style Guide

## Naming Conventions

### Variables

Arrays should typically be plural nouns (e.g. `dirs`, `dists`), while other variable should be single nouns. This extends to boolean mask arrays, which should still be names as plutal nouns (e.g. `valids` instead of `is_valid`).
Index arrays (containing pairs of ($i$, $j$) indices) should be named with `_ijs` suffix (e.g. `tofill_ijs`).

#### Standard variable names

Some variables are used so frequently that they should be named consistently across the codebase. These include:

- `dir_scheme`: instance of `D8Directions` defining the meaning of code numbers in flow direction arrays
- `dirs`: flow direction array
- `dists`: distance array
- `valids`: boolean mask array indicating valid cells
- `indegs`: indegree array for flow direction grid

### Functions

Function names should use snake_case, and is generally prefixed with a verb describing the action (e.g. `compute_`, `label_`).

## Documentation

### Docstrings

All public functions should have descriptive docstrings written in the *NumPy style*. This includes:

- A clear, concise summary line.
- A **Parameters** section detailing variable names, types (e.g. `NDArray[int]`), and default values if optional.
  - Each parameter should start with the short phrase describing the parameter, without the leading 'A' or 'The', and end without a period.
  - More detailed descriptions of parameters can be provided in a separate paragraph after the short phrase.
  - For optional parameters, the default value should be specified at the end of the description in parentheses (e.g. 'The default backend is `"python"`').
- A **Returns** section specifying returned values, types, and descriptions of output structures.
  - The documentation should be in the same style as that for the Parameters section.
- A **Raises** section detailing exceptions raised and under what conditions.

### Type Annotations

Functions should be fully type-annotated:

- Standard type hints from typing (e.g. `Optional`, `Literal`) should be used when warranted (e.g. `tuple` if preferred over `Tuple` from the `typing` package).
- NumPy arrays should use `numpy.typing.NDArray` with the inner type specified (e.g. `npt.NDArray[np.integer]`, `npt.NDArray[np.number]`).

----
Last updated: Jul 12, 2026 ([@williameclee](https://github.com/williameclee))
