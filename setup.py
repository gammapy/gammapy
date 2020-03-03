# Licensed under a 3-clause BSD style license - see LICENSE.rst
import sys
import setuptools
from distutils.version import LooseVersion

if LooseVersion(setuptools.__version__) < "30.3":
    sys.stderr.write("ERROR: setuptools 30.3 or later is required by gammapy\n")
    sys.exit(1)

# TODO: check if setuptools_scm, numpy, ... are OK
# Exit with good error message telling people to install those first if not


from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np


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
