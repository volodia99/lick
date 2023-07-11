import sys
from distutils.extension import Extension

import numpy
from Cython.Build import cythonize
from setuptools import setup

if sys.version_info >= (3, 9):
    # keep in sync with runtime requirements (pyproject.toml)
    define_macros = [("NPY_TARGET_VERSION", "NPY_1_18_API_VERSION")]
else:
    define_macros = []

setup(
    ext_modules=cythonize(
        [
            Extension(
                "lick._vendor.vectorplot.core",
                ["src/lick/_vendor/vectorplot/core.pyx"],
                include_dirs=[numpy.get_include()],
                define_macros=define_macros,
            ),
        ],
        compiler_directives={"language_level": 3},
        annotate=True,
    ),
)
