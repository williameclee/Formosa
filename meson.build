project('formosa', 'fortran')

py = import('python').find_installation(pure: false)

# Where platform-specific files (extensions) should go
pkg_pure_dir      = py.get_install_dir(pure: false)
pkg_geom_plat_dir = py.get_install_dir(pure: false, subdir: 'formosa/geomorphology')

# Install pure Python package
install_subdir('src/formosa', install_dir: pkg_pure_dir)

# Fortran extension
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
    '--f90flags="-march=native -fopenmp -O3"', # -pg for profiling
    '-lgomp',
  ],
  install: true,
  install_dir: pkg_geom_plat_dir,
  build_by_default: true,
)