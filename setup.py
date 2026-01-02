from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
from pathlib import Path


extensions = [
    Extension(
        "formosa.flow_distance_loop",  # Python module name
        ["src/formosa/flow_distance_loop.pyx"],
        include_dirs=[np.get_include()],  # required for numpy headers
        extra_compile_args=[],
        extra_link_args=[],
    ),
    Extension(
        "formosa.away_from_high_loop",  # Python module name
        ["src/formosa/away_from_high_loop.pyx"],
        include_dirs=[np.get_include()],  # required for numpy headers
        extra_compile_args=[],
        extra_link_args=[],
    ),
    Extension(
        "formosa.towards_low_loop",  # Python module name
        ["src/formosa/towards_low_loop.pyx"],
        include_dirs=[np.get_include()],  # required for numpy headers
        extra_compile_args=[],
        extra_link_args=[],
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        language_level="3",
        annotate=False,  # set to True if you want annotated HTML
    )
)
