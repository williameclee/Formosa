project('formosa', 'c', 'cython', 'fortran')

py = import('python').find_installation(pure: false)
py_dep = py.dependency()

# NumPy include dir (needed for Cython/NumPy C-API)
# numpy_inc = run_command(
#   py, ['-c', 'import numpy; print(numpy.get_include())'],
#   check: true
# ).stdout().strip()
# numpy_incdir = include_directories(numpy_inc)
numpy_inc = run_command(
  py, ['-c', 'import numpy; print(numpy.get_include())'],
  check: true
).stdout().strip()

# FIX: If numpy_inc is inside the project, make it relative
source_root = meson.project_source_root()
if numpy_inc.startswith(source_root)
  # Strip the root path + leading slash to make it relative
  numpy_inc = numpy_inc.replace(source_root + '/', '')
endif

numpy_incdir = include_directories(numpy_inc)

# Where platform-specific files (extensions) should go
pkg_pure_dir      = py.get_install_dir(pure: false)
pkg_geom_plat_dir = py.get_install_dir(pure: false, subdir: 'formosa/geomorphology')

# --- Install your pure Python package (src layout)
install_subdir('src/formosa', install_dir: pkg_pure_dir)

# --- Cython extension(s)
py.extension_module(
  'flow_distance_loop',
  'src/formosa/geomorphology/flow_distance_loop.pyx',
  include_directories: [numpy_incdir],
  dependencies: [py_dep],
  install: true,
  subdir: 'formosa/geomorphology',
)

# --- f2py extension (build via numpy.f2py, then install into the package dir)
ext_suffix = run_command(
  py, ['-c', 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX") or "")'],
  check: true
).stdout().strip()

flowdir_f = custom_target(
  'flowdir_f',
  input: files('src/formosa/geomorphology/flowdir.f95'),
  output: ['flowdir_f' + ext_suffix],
  command: [
    py.full_path(), '-m', 'numpy.f2py',
    '-c', '@INPUT@',
    '-m', 'flowdir_f',
    # OpenMP flags
    '--fcompiler=gnu95',
    '--f90flags="-march=native -fopenmp -O3"',
    '-lgomp',
  ],
  build_by_default: true,
  install: true,
  install_dir: pkg_geom_plat_dir,
)