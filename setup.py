# Licensed under a 3-clause BSD style license - see LICENSE.rst
from distutils.extension import Extension

import numpy as np
import setuptools
from Cython.Build import cythonize


def make_cython_extension(filename):
    return Extension(
        filename.strip(".pyx").replace("/", "."),
        [filename],
        include_dirs=[np.get_include()],
    )


cython_files = [
    "gammapy/stats/fit_statistics_cython.pyx",
]

ext_modules = cythonize([make_cython_extension(_) for _ in cython_files])

setuptools.setup(use_scm_version=True, ext_modules=ext_modules)
