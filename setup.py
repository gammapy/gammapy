#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

# NOTE: The configuration for the package, including the name, version, and
# other information are set in the setup.cfg file.
import sys

# First provide helpful messages if contributors try and run legacy commands
# for tests or docs.

TEST_HELP = """
Note: running tests is no longer done using 'python setup.py test'. Instead
you will need to run:

    tox -e test

If you don't already have tox installed, you can install it with:

    pip install tox

If you only want to run part of the test suite, you can also use pytest
directly with::

    pip install -e .[test]
    pytest

For more information, see:

  http://docs.astropy.org/en/latest/development/testguide.html#running-tests
"""

if "test" in sys.argv:
    print(TEST_HELP)
    sys.exit(1)

DOCS_HELP = """
Note: building the documentation is no longer done using
'python setup.py build_docs'. Instead you will need to run:

    tox -e build_docs

If you don't already have tox installed, you can install it with:

    pip install tox

You can also build the documentation with Sphinx directly using::

    pip install -e .[docs]
    cd docs
    make html

For more information, see:

  http://docs.astropy.org/en/latest/install.html#builddocs
"""

if "build_docs" in sys.argv or "build_sphinx" in sys.argv:
    print(DOCS_HELP)
    sys.exit(1)


# imports here so that people get the nice error messages above without needing
# build dependencies
import numpy as np  # noqa: E402
from Cython.Build import cythonize  # noqa: E402
from setuptools import Extension, setup  # noqa: E402

kwargs = dict(
    include_dirs=[np.get_include()],
    define_macros=[
        # fixes a warning when compiling
        ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
        # defines the oldest numpy we want to be compatible with
        ("NPY_TARGET_VERSION", "NPY_1_21_API_VERSION"),
    ],
)

extensions = [
    Extension(
        "gammapy.stats.fit_statistics_cython",
        sources=["gammapy/stats/fit_statistics_cython.pyx"],
        **kwargs,
    ),
]

setup(
    ext_modules=cythonize(extensions),
)
