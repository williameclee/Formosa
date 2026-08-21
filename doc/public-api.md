# Public API Inventory

This document is a quick map of Formosa's supported public API. It is intended to help users find the right operation and to help understand backend support and test coverage before changing code. Function signatures and detailed parameter documentation remain authoritative in the source docstrings.

## How to Read This Inventory

The symbol in each table links to its implementation. The import path is kept as plain code so it can be copied. Test labels link to the most relevant test file; they describe coverage found during the audit, not a formal coverage guarantee.

### Backend Labels

- **Fortran + Python**: The public function accepts either backend.
- **Fortran only**: The operation requires the native extension or exposes no Python alternative.
- **Python only**: The operation is implemented in Python and backend selection is not exposed.
- **Mixed**: A high-level operation combines Python orchestration with native computational stages.
- **N/A**: A backend distinction is not meaningful, typically for I/O, containers, configuration, or plotting.

Unless stated otherwise, functions supporting backend selection default to `backend="fortran"`.

### Test Labels

- **Direct**: Tests call the public symbol explicitly.
- **Parity**: Tests exercise both Fortran and Python backends.
- **Indirect**: Tests exercise the behaviour through another public API.
- **Partial**: Some important paths are tested, but the complete public behaviour is not.
- **None**: No test exercising the public behaviour was identified.

## Primary User API

These symbols are available directly from `formosa`.

| Symbol | Import path | Purpose | Backend | Tests | Notes |
| --- | --- | --- | --- | --- | --- |
| [`DEMGrid`](../src/formosa/dem/terrain_grid.py) | `.DEMGrid` | Stores a DEM, coordinates, masks, and lazily derived terrain and drainage products. | Mixed | [Direct](../tests/test_demgrid.py) | Preferred workflow-oriented interface; see the method summary below. |
| [`D8Directions`](../src/formosa/geomorphology/drainage/directions.py) | `.D8Directions` | Defines D8 direction codes, offsets, distances, and the no-flow code. | N/A | Indirect | Pass instances as `dir_scheme`; the default scheme is created at function definition time. |
| [`read_dem`](../src/formosa/dem/demio.py) | `.read_dem` | Reads one raster band and returns a `DEMGrid`. | N/A | [Direct](../tests/test_demgrid.py) | Uses Rasterio; `nan_value` controls replacement of raster nodata values. |
| [`gmrt`](../src/formosa/dem/api/gmrt.py) | `.gmrt` | Downloads and optionally caches a DEM from GMRT. | N/A | None | Performs network and optional filesystem I/O; returns `Z`, `X`, `Y`, and an affine transform. |
| [`opentopo`](../src/formosa/dem/api/opentopo.py) | `.opentopo` | Downloads and optionally caches a global DEM from OpenTopography. | N/A | None | Requires an API key; returns `Z`, `X`, `Y`, and an affine transform. |

### Important `DEMGrid` Members

`DEMGrid` caches many derived properties. Mutating the DEM or validity masks through future APIs must invalidate dependent caches consistently.

| Member | Purpose | Tests | Notes |
| --- | --- | --- | --- |
| `shape`, `slope` | Report grid dimensions and derive terrain slope. | Partial | `slope` is computed lazily. |
| `ocean_mask`, `sea_mask` | Identify ocean/sea cells and expose their masks. | [Direct](../tests/test_demgrid.py) | Ocean invalidation clears dependent cached products. |
| `flowdir`, `indegree` | Derive D8 flow directions and upstream-cell counts. | Indirect | Native computational paths are used internally. |
| `accumulation`, `strahler_order` | Derive drainage accumulation and Strahler order. | Indirect | Depend on the cached flow-direction grid. |
| `dist2source`, `dist2sink`, `flow_distance` | Derive along-flow distance products. | None | Coordinate arrays determine whether distances are geometric or grid based. |
| `watersheds` | Label drainage basins. | Indirect | Depends on flow directions and the current validity mask. |
| `fill_depressions()` | Fill enclosed depressions and update the stored DEM. | [Direct](../tests/test_demgrid.py) | Chainable; preserves boundary outlets. |
| `invalidate_ocean_basins()` | Invalidate boundary-connected ocean basins. | [Direct](../tests/test_demgrid.py) | Chainable and designed to be idempotent. |
| `ridgedir`, `dist2ridge`, `ridge_strahler_order` | Derive ridge-network directions, distances, and orders. | Partial | Primarily backed by native routines. |

## DEM Utilities

| Symbol | Import path | Purpose | Backend | Tests | Notes |
| --- | --- | --- | --- | --- | --- |
| [`transform2xy`](../src/formosa/dem/utils.py) | `.dem.transform2xy` | Builds cell-centre `X` and `Y` coordinate arrays from a Rasterio affine transform and raster shape. | N/A | Indirect | Returns two 2-D arrays with the requested shape. |

`set_data_dir` exists in `formosa.core`, but it is not currently exported by an `__all__` and is therefore outside this inventory's public boundary.

## Terrain and Drainage

The shorter paths under `formosa.geomorphology` are recommended for the core operations listed there. Lower-level drainage operations are available from `formosa.geomorphology.drainage`.

| Symbol | Import path | Purpose | Backend | Tests | Notes |
| --- | --- | --- | --- | --- | --- |
| [`compute_slope`](../src/formosa/geomorphology/terrain.py) | `.geomorphology.compute_slope` | Computes terrain slope from a DEM using coordinates or explicit cell spacing. | Python only | None | Supply either `x`/`y` or `dx`/`dy` as required by the implementation. |
| [`compute_isolation`](../src/formosa/geomorphology/terrain.py) | `.geomorphology.terrain.compute_isolation` | Calculates terrain isolation, isolation limit points, and boundary censoring. | Fortran only | [Direct](../tests/test_terrain_f.py) | Returns `(isos, ilpis, ilpjs, censored)`; supports custom `dx`/`dy` spacing and validity masks. |
| [`compute_prominence`](../src/formosa/geomorphology/terrain.py) | `.geomorphology.terrain.compute_prominence` | Computes topographic prominence for all valid cells in a DEM. | Fortran only | [Direct](../tests/test_terrain_f.py) | Returns `proms`; supports direction schemes and validity masks. |
| [`get_neighbour_values`](../src/formosa/geomorphology/drainage/neighbours.py) | `.geomorphology.get_neighbour_values` | Collects directional neighbour values for every grid cell. | Python only | Indirect | Padding and optional inclusion of the centre cell are configurable. |
| [`compute_downstream_indices`](../src/formosa/geomorphology/drainage/neighbours.py) | `.geomorphology.drainage.compute_downstream_indices` | Converts a direction grid into downstream indices. | Python only | Indirect | Can return flat indices or row/column indices and optionally validate directions. |
| [`detect_ocean_basins_from_boundary`](../src/formosa/geomorphology/drainage/preprocessing.py) | `.geomorphology.detect_ocean_basins_from_boundary` | Labels threshold-matching basins connected to the raster boundary. | Fortran only | [Direct](../tests/test_preprocessing_f.py) | Supports exact-level and flood-below modes. |
| [`invalidate_ocean_basins`](../src/formosa/geomorphology/drainage/preprocessing.py) | `.geomorphology.invalidate_ocean_basins` | Returns a validity mask with sufficiently large ocean basins invalidated. | Fortran only | [Direct](../tests/test_preprocessing_f.py) | `min_size` is inclusive. |
| [`fill_depressions`](../src/formosa/geomorphology/drainage/preprocessing.py) | `.geomorphology.fill_depressions` | Fills enclosed DEM depressions while respecting outlets and invalid cells. | Fortran only | [Direct](../tests/test_preprocessing_f.py) | Does not mutate the input; `max_fill_size` can limit filled regions. |
| [`compute_flowdir`](../src/formosa/geomorphology/drainage/flowdir.py) | `.geomorphology.compute_flowdir` | Computes D8 flow directions, optionally filling depressions and resolving flats. | Mixed | Indirect | Returns `(dirs, flats, synthetic_gradients)`; no public backend selector. |
| [`count_indegree`](../src/formosa/geomorphology/drainage/flowdir.py) | `.geomorphology.count_indegree` | Counts upstream neighbours for each valid cell. | Fortran + Python | [Parity](../tests/test_flowdir.py) | Accepts an optional validity mask. |
| [`find_acyclic_flowdirs`](../src/formosa/geomorphology/drainage/flowdir.py) | `.geomorphology.drainage.find_acyclic_flowdirs` | Marks valid cells that are not part of a directed flow cycle. | Fortran + Python | [Parity](../tests/test_flowdir.py) | Optional precomputed in-degrees avoid duplicate work. |
| [`find_cyclic_flowdirs`](../src/formosa/geomorphology/drainage/flowdir.py) | `.geomorphology.drainage.find_cyclic_flowdirs` | Marks valid cells that belong to directed flow cycles. | Fortran + Python | [Parity](../tests/test_flowdir.py) | Defined as the valid complement of the acyclic result. |
| [`compute_flow_accumulation`](../src/formosa/geomorphology/drainage/metrics.py) | `.geomorphology.compute_flow_accumulation` | Accumulates upstream cell counts or supplied weights. | Fortran + Python | Partial | Optional in-degrees and downstream indices can be reused. |
| [`compute_flow_strahler_order`](../src/formosa/geomorphology/drainage/metrics.py) | `.geomorphology.compute_flow_strahler_order` | Computes cell-level Strahler stream order. | Fortran + Python | [Parity](../tests/test_metrics_parity.py) | Masks and optionally supplied in-degrees are covered by parity tests. |
| [`compute_dist2source`](../src/formosa/geomorphology/drainage/metrics.py) | `.geomorphology.compute_dist2source` | Computes along-flow distance from upstream sources. | Fortran only | None | Uses geometric distance when `x` and `y` are supplied. |
| [`compute_dist2sink`](../src/formosa/geomorphology/drainage/metrics.py) | `.geomorphology.compute_dist2sink` | Computes along-flow distance to downstream sinks. | Fortran only | None | Uses geometric distance when `x` and `y` are supplied. |
| [`label_watersheds`](../src/formosa/geomorphology/drainage/watersheds.py) | `.geomorphology.label_watersheds` | Labels cells by their terminal drainage basin. | Fortran + Python | Indirect | Returns an integer label grid and respects the validity mask. |
| [`compute_dist2conf_max`](../src/formosa/geomorphology/drainage/ridges.py) | `.geomorphology.compute_dist2conf_max` | Computes maximum branch distance to a confluence. | Fortran only | [Direct](../tests/test_ridges_f.py) | Coordinate arrays enable geometric rather than grid distance. |
| [`compute_ridgedir`](../src/formosa/geomorphology/drainage/ridges.py) | `.geomorphology.compute_ridgedir` | Derives ridge directions from a drainage direction grid. | Fortran only | Indirect | Produces a direction grid suitable for other ridge operations. |
| [`compute_dist2ridge`](../src/formosa/geomorphology/drainage/ridges.py) | `.geomorphology.compute_dist2ridge` | Computes distance from each cell to its associated ridge. | Fortran only | Indirect | `dir_is_ridge=True` treats the input as an existing ridge direction grid. |
| [`compute_ridge_strahler_order`](../src/formosa/geomorphology/drainage/ridges.py) | `.geomorphology.compute_ridge_strahler_order` | Computes Strahler order on the derived or supplied ridge network. | Fortran + Python | [Parity](../tests/test_ridges.py) | Forwards masks to the selected backend. |

The flat-resolution functions in `drainage/flat_resolution.py` are currently implementation details: they are not included in a package `__all__`, even though their names do not begin with an underscore.

## Flow-network API

Flow graphs use parallel arrays: per-arc orders, a vertex coordinate/index array, and per-arc endpoint ranges into that vertex array. Check individual docstrings for accepted orientations and return layouts.

| Symbol | Import path | Purpose | Backend | Tests | Notes |
| --- | --- | --- | --- | --- | --- |
| [`construct_flowgraph`](../src/formosa/geomorphology/drainage/network/construction.py) | `.geomorphology.construct_flowgraph` | Constructs ordered graph arcs from a flow-direction raster. | Fortran + Python | [Parity](../tests/test_network_construction_parity.py) | Detects directed cycles and validates that selected flow edges are represented. |
| [`create_flowline_plot_data`](../src/formosa/geomorphology/drainage/network/construction.py) | `.geomorphology.drainage.create_flowline_plot_data` | Converts flow directions into line coordinates for plotting. | Python only | None | This is plotting data, not the ordered flow-graph representation. |
| [`concat_flowgraph`](../src/formosa/geomorphology/drainage/network/editing.py) | `.geomorphology.concat_flowgraph` | Concatenates same-order arcs with NaN separators for efficient drawing. | Python only | [Direct](../tests/test_network_editing.py) | Intended mainly to reduce plotting calls. |
| [`insert_endpt`](../src/formosa/geomorphology/drainage/network/editing.py) | `.geomorphology.drainage.network.insert_endpt` | Splits an arc by turning an interior vertex into an endpoint. | Python only | [Direct](../tests/test_network_editing.py) | Can optionally compact unused vertices. |
| [`remove_unused_vertices`](../src/formosa/geomorphology/drainage/network/editing.py) | `.geomorphology.drainage.network.remove_unused_vertices` | Compacts vertices not referenced by any arc. | Python only | [Direct](../tests/test_network_editing.py) | Rewrites endpoint indices to match the compacted array. |
| [`locate_invalid_graph_topology`](../src/formosa/geomorphology/drainage/network/validation.py) | `.geomorphology.drainage.network.locate_invalid_graph_topology` | Finds disallowed segment intersections within and between graph arcs. | Fortran + Python | [Parity](../tests/test_network_validation.py) | Native overflow/retry behaviour has additional focused tests. |
| [`simplify_flowgraph`](../src/formosa/geomorphology/drainage/network/simplification.py) | `.geomorphology.simplify_flowgraph` | Simplifies one or several flow graphs with the Ramer-Douglas-Peucker algorithm. | Fortran only currently | [Direct](../tests/test_network_simplification_f.py) | Accepts `backend="python"`, but that path currently raises `NotImplementedError`; multi-graph input aligns overlaps first. |

`find_graph_overlaps` and `solve_graph_overlaps` have direct tests but are not currently exported from `formosa.geomorphology.drainage.network`; they remain outside the declared public boundary.

### Public Graph Exceptions

| Exception | Import path | Meaning |
| --- | --- | --- |
| [`GraphTopologyError`](../src/formosa/geomorphology/drainage/network/validation.py) | `.geomorphology.drainage.GraphTopologyError` | Base class for graph topology validation failures. |
| [`DirectedFlowCycleError`](../src/formosa/geomorphology/drainage/network/validation.py) | `.geomorphology.drainage.DirectedFlowCycleError` | The selected flow field contains one or more directed cycles. |
| [`IncompleteFlowGraphError`](../src/formosa/geomorphology/drainage/network/validation.py) | `.geomorphology.drainage.IncompleteFlowGraphError` | Graph construction omitted one or more selected directed edges. |
| [`InvalidOriginalGraphTopology`](../src/formosa/geomorphology/drainage/network/validation.py) | `.geomorphology.drainage.InvalidOriginalGraphTopology` | An invalid result originated from already-invalid input topology. |
| [`UnresolvedSimplificationTopology`](../src/formosa/geomorphology/drainage/network/validation.py) | `.geomorphology.drainage.UnresolvedSimplificationTopology` | Simplification introduced or failed to resolve invalid topology from valid input. |

## Geometry

All public geometry predicates accept `backend="fortran"` or `backend="python"` and default to Fortran. They validate points at the public wrapper and have backend-parity tests.

| Symbol | Import path | Purpose | Backend | Tests | Notes |
| --- | --- | --- | --- | --- | --- |
| [`orient`](../src/formosa/geomorphology/geometry/intersections.py) | `.geomorphology.geometry.orient` | Returns the signed orientation determinant of 3 2-D points. | Fortran + Python | [Parity](../tests/test_geometry_intersections.py) | Python integer inputs use exact arithmetic; the Fortran integer overloads use widened intermediates and same-kind saturating results. |
| [`incircle`](../src/formosa/geomorphology/geometry/intersections.py) | `.geomorphology.geometry.incircle` | Returns the signed in-circle determinant of 4 2-D points. | Fortran + Python | [Parity](../tests/test_geometry_intersections.py) | `oriented=True` normalizes the sign for either triangle winding; Fortran integer overloads preserve the input kind and saturate on result overflow. |
| [`on_segment`](../src/formosa/geomorphology/geometry/intersections.py) | `.geomorphology.geometry.on_segment` | Tests whether a point lies on a closed line segment. | Fortran + Python | [Parity](../tests/test_geometry_intersections.py) | Endpoints count as on the segment. |
| [`bboxes_overlap`](../src/formosa/geomorphology/geometry/intersections.py) | `.geomorphology.geometry.bboxes_overlap` | Tests whether two closed segment bounding boxes overlap. | Fortran + Python | [Parity](../tests/test_geometry_intersections.py) | Bounding-box overlap does not itself imply segment intersection. |
| [`lines_intersect`](../src/formosa/geomorphology/geometry/intersections.py) | `.geomorphology.geometry.lines_intersect` | Classifies intersection between two closed 2-D line segments. | Fortran + Python | [Parity](../tests/test_geometry_intersections.py) | Used by flow-graph topology validation. |

## Meshing

The meshing API triangulates 2-D vertices and recovers edges of a planar
straight-line graph. These functions are available from
`formosa.geomorphology.meshing.triangulation`; constraint validation is
available from `formosa.geomorphology.meshing.validation`.

| Symbol | Import path | Purpose | Backend | Tests | Notes |
| --- | --- | --- | --- | --- | --- |
| [`triangulate_points`](../src/formosa/geomorphology/meshing/triangulation.py) | `.geomorphology.meshing.triangulation.triangulate_points` | Computes a canonical unconstrained Delaunay triangulation of 2-D vertices. | Fortran + Python | [Parity](../tests/test_triangulation.py) | Returns counterclockwise triangles in deterministic lexicographic order; the Fortran backend requires `int32`-representable integer coordinates. |
| [`find_facet_neighbours`](../src/formosa/geomorphology/meshing/triangulation.py) | `.geomorphology.meshing.triangulation.find_facet_neighbours` | Finds the adjacent triangle across each side of every triangle. | Fortran + Python | [Parity](../tests/test_triangulation.py) | Uses `-1` for mesh-boundary sides. |
| [`flip_quadrilateral_edge`](../src/formosa/geomorphology/meshing/triangulation.py) | `.geomorphology.meshing.triangulation.flip_quadrilateral_edge` | Replaces the diagonal of a convex pair of adjacent triangles. | Fortran + Python | [Parity](../tests/test_triangulation.py) | Does not mutate the supplied triangles or optional neighbour array; defaults to the Python backend. |
| [`recover_constraint_edge`](../src/formosa/geomorphology/meshing/triangulation.py) | `.geomorphology.meshing.triangulation.recover_constraint_edge` | Recovers one constraint as a mesh edge using legal edge flips. | Fortran + Python | [Parity](../tests/test_triangulation.py) | Can preserve an explicit set of locked mesh edges. |
| [`recover_constraint_edges`](../src/formosa/geomorphology/meshing/triangulation.py) | `.geomorphology.meshing.triangulation.recover_constraint_edges` | Recovers non-crossing constraints sequentially while preserving earlier constraints. | Fortran + Python | [Parity](../tests/test_triangulation.py) | Reports the position and canonical vertex pair of a constraint that cannot be recovered. |
| [`validate_constraints`](../src/formosa/geomorphology/meshing/validation.py) | `.geomorphology.meshing.validation.validate_constraints` | Validates the normalisation, bounds, boundary coverage, and intersections of a constraint graph. | Fortran + Python | Indirect | Called by `ConstraintGraph.validate`; intersection checks use the selected backend. |

## Graphics

| Symbol | Import path | Purpose | Backend | Tests | Notes |
| --- | --- | --- | --- | --- | --- |
| [`hillshade`](../src/formosa/graphics/hillshade.py) | `.graphics.hillshade` | Computes shaded relief from a 2-D elevation array. | N/A | None | Supports the implementation's hard and soft shading methods. |
| [`light_terrain`](../src/formosa/graphics/colour.py) | `.graphics.light_terrain` | Creates a light terrain Matplotlib colormap. | N/A | None | Supports truncation and reversal. |
| [`dune`](../src/formosa/graphics/colour.py) | `.graphics.dune` | Creates a dune-toned Matplotlib colormap. | N/A | None | Supports an optional alpha channel. |
| [`mist`](../src/formosa/graphics/colour.py) | `.graphics.mist` | Creates a mist-toned Matplotlib colormap. | N/A | None | Supports truncation and reversal. |
| [`iceberg`](../src/formosa/graphics/colour.py) | `.graphics.iceberg` | Creates an iceberg-toned Matplotlib colormap. | N/A | None | Supports truncation and reversal. |

`truncate_colormap` is used by these factories but is not exported in `formosa.graphics.__all__`.

## Backend Infrastructure

`formosa.utils` exports `BACKENDS`, the `Backend` type alias, and `raise_fortran_error`. These are useful to contributors implementing wrappers, but most users should select backends through a public function's `backend` argument rather than call backend infrastructure directly.

The native extension is built as `formosa.geomorphology._native`. Modules under `_backends` and the extension's generated module namespaces are internal and must not be treated as stable import paths.

## Known Documentation and Coverage Gaps

- `gmrt` and `opentopo` have no dedicated tests for validation, caching, request construction, or response handling.
- `compute_slope`, graphics functions, flow-distance functions, and `create_flowline_plot_data` have no direct tests identified in this audit.
- Several `DEMGrid` derived properties are exercised only indirectly or have no focused cache-invalidation tests.
- `simplify_flowgraph` exposes a Python backend option even though that backend is not implemented.
- Some useful functions have public-looking names but are not exported through `__all__`. Promote them deliberately rather than relying on incidental module imports.

## Maintenance Guidance

When adding, removing, or changing a public symbol:

1. Export it deliberately through the appropriate `__all__`.
2. Add or update its row here using the supported import path.
3. Link the symbol to the source file using a relative repository link; avoid line anchors because they become stale quickly.
4. Determine backend support from the public wrapper, not merely from similarly named internal routines.
5. Add direct tests. Add parity tests whenever both Fortran and Python backends are supported.
6. Link the most representative tests rather than every test touching the implementation.
7. Record array orientation, mutation, caching, I/O, and important exceptions when they affect safe use.
8. Mark renamed or removed import paths as compatibility changes in release notes or migration documentation.

To re-audit the declared public boundary, inspect all package `__init__.py` files and compare their `__all__` values with this document. Then search the test suite for direct calls to each symbol and for parameterisation over `backend`.

---

Maintainer: [En-Chi Lee (`@williameclee`)](https://github.com/williameclee)

Last updated: 2026-08-21
