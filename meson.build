project('formosa', 'fortran', version: '0.1.0a0')

py = import('python').find_installation(pure: false)

# Where platform-specific files (extensions) should go
pkg_pure_dir = py.get_install_dir(pure: false)
pkg_geom_plat_dir = py.get_install_dir(pure: false, subdir: 'formosa/geomorphology')

# Install pure Python package
install_subdir('src/formosa', install_dir: pkg_pure_dir)

# Fortran extension
ext_suffix = run_command(
  py,
  [
    '-c', 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX") or "")',
  ],
  check: true,
).stdout().strip()

custom_target(
  'drainage_f',
  input: files(
    'src/formosa/geomorphology/drainage/utils.f90',
    'src/formosa/geomorphology/geometry/geometry.f90',
    'src/formosa/geomorphology/geometry/intersections.f90',
    'src/formosa/geomorphology/drainage/_backends/preprocessing.f90',
    'src/formosa/geomorphology/drainage/network/_backends/construction.f90',
    'src/formosa/geomorphology/drainage/network/_backends/validation.f90',
    'src/formosa/geomorphology/drainage/network/_backends/simplification.f90',
    'src/formosa/geomorphology/drainage/_backends/basins.f90',
    'src/formosa/geomorphology/drainage/_backends/metrics.f90',
    'src/formosa/geomorphology/drainage/_backends/flowdir_flat_resolution.f90',
    'src/formosa/geomorphology/drainage/_backends/flowdir.f90',
    'src/formosa/geomorphology/drainage/_backends/ridges.f90',
  ),
  output: ['drainage_f' + ext_suffix],
  command: [
    py.full_path(),
    '-m', 'numpy.f2py',
    '-m', 'drainage_f',
    # OpenMP flags
    '--f90flags="-march=native -fopenmp -O3"', # -pg for profiling
    '-lgomp',
    '-c', '@INPUT@',
  ],
  install: true,
  install_dir: pkg_geom_plat_dir,
  build_by_default: true,
)