# Formosa

Formosa is an experimental Python package for terrain analysis, drainage network extraction, and cartographic visualisation. It has a compiled Fortran routines for computationally intensive geomorphology operations.

## Features

- Read digital elevation models supported by Rasterio, including GeoTIFF and SRTM HGT rasters.
- Download elevation data from GMRT and OpenTopography.
- Represent a DEM and lazily derive terrain and drainage products with `DEMGrid`.
- Fill depressions, resolve flats, and compute D8 flow directions.
- Compute flow accumulation, Strahler order, watersheds, flow distances, and ridge metrics.
- Construct, edit, and simplify drainage-network graphs.
- Use matching Fortran and Python implementations for selected operations.
- Produce hillshade and terrain-oriented Matplotlib colourmaps.

See the [public API inventory](doc/public-api.md) for supported import paths, backend availability, test coverage, and implementation links.

## Requirements

- Python 3.12 or newer
- A Fortran compiler supported by NumPy's F2PY and Meson
- OpenMP and `libgomp` for the native extension

The current native build passes GCC-specific optimisation and OpenMP flags, so other compiler toolchains may require build configuration changes.

## Installation from Source

Clone the repository and install it into an environment with a working Fortran toolchain:

```console
python -m pip install .
```

For development with [uv](https://docs.astral.sh/uv/):

```console
uv sync
```

Both commands compile the native extension. Formosa is not currently documented here as a published package; use the repository as the installation source unless a release says otherwise.

## Quick Start

```python
import numpy as np

from formosa import DEMGrid
from formosa.geomorphology import compute_flowdir

# DEMGrid accepts a raster path or an array. Supply coordinates for an array.
dem = np.array(
    [
        [8.0, 7.0, 6.0, 5.0],
        [7.0, 2.0, 2.0, 4.0],
        [6.0, 2.0, 1.0, 3.0],
        [5.0, 4.0, 3.0, 2.0],
    ],
    dtype=np.float32,
)
x, y = np.meshgrid(np.arange(dem.shape[1]), np.arange(dem.shape[0]))

grid = DEMGrid(dem, x=x, y=y)
grid.fill_depressions()

directions, flat_cells, synthetic_gradient = compute_flowdir(
    grid.dem, valids=grid.valid,
)
```

For raster input, `DEMGrid(path)` reads the raster and derives its coordinates and affine transform:

```python
from formosa import DEMGrid

grid = DEMGrid("path/to/elevation.tif")
print(grid.shape)
print(grid.slope)
```

## Backends

Several public functions accept `backend="fortran"` or `backend="python"` and default to Fortran. Other functions are native-only, Python-only, or combine multiple implementation stages without exposing backend selection. Do not import generated extension namespaces or modules under `_backends`; those are internal implementation details.

The [public API inventory](doc/public-api.md#backend-labels) records backend support per function. Backend parity is tested where both implementations are available, but not every public operation currently has both backends.

## Development and Testing

Run the test suite from the repository root:

```console
uv run pytest tests/
```

Build a wheel with:

```console
uv build --wheel
```

Before changing a public API:

1. Update the appropriate package `__all__` deliberately.
2. Add direct tests and backend-parity tests where applicable.
3. Update the [public API inventory](doc/public-api.md).
4. Follow the project [style guide](STYLE_GUIDE.md).

## Documentation

- [Style guide](STYLE_GUIDE.md) — project code and documentation conventions
- [Public API inventory](doc/public-api.md) — functions, supported imports, backends, tests, and maintenance notes

Detailed signatures and parameter descriptions currently live in source docstrings. The API inventory is a navigation and maintenance aid rather than a replacement for those docstrings.

---

Maintainer: [En-Chi Lee (`@williameclee`)](https://github.com/williameclee)

Last updated: 2026-08-10
