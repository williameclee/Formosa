# Style Guide

## Naming Conventions

### Variables

Arrays should typically be plural nouns (e.g. `dirs`, `dists`), while other variable should be single nouns. This extends to boolean mask arrays, which should still be names as plutal nouns (e.g. `valids` instead of `is_valid`).
Index arrays (containing pairs of (i, j) indices) should be named with `_ijs` suffix (e.g. `tofill_ijs`).

#### Standard variable names

Some variables are used so frequently that they should be named consistently across the codebase. These include:

- `dir_scheme`: instance of `D8Directions` defining the meaning of code numbers in flow direction arrays
- `dirs`: flow direction array
- `dists`: distance array
- `valids`: boolean mask array indicating valid cells
- `indegs`: indegree array for flow direction grid

### Functions

Function names should use snake_case, and is generally prefixed with a verb describing the action (e.g. `compute_`, `label_`).

----
Last updated: Jun 11, 2026 ([@williameclee](https://github.com/williameclee))
