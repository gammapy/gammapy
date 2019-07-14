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

include_path = [np.get_include()]

# TODO: simplify this with a helper function?
ext_modules = [
    Extension(
        "gammapy.detect._test_statistics_cython",
        ["gammapy/detect/_test_statistics_cython.pyx"],
        include_path,
    ),
    Extension("gammapy.maps._sparse", ["gammapy/maps/_sparse.pyx"], include_path),
    Extension(
        "gammapy.stats.fit_statistics_cython",
        ["gammapy/stats/fit_statistics_cython.pyx"],
        include_path,
    ),
]

ext_modules = cythonize(ext_modules)

setuptools.setup(use_scm_version=True, ext_modules=ext_modules)
