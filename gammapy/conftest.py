# Licensed under a 3-clause BSD style license - see LICENSE.rst
# this contains imports plugins that configure py.test for astropy tests.
# by importing them here in conftest.py they are discoverable by py.test
# no matter how it is invoked within the source tree.
import os
from . import version

from astropy.version import version as astropy_version

if astropy_version < "3.0":
    # With older versions of Astropy, we actually need to import the pytest
    # plugins themselves in order to make them discoverable by pytest.
    from astropy.tests.pytest_plugins import *
else:
    # As of Astropy 3.0, the pytest plugins provided by Astropy are
    # automatically made available when Astropy is installed. This means it's
    # not necessary to import them here, but we still need to import global
    # variables that are used for configuration.
    from astropy.tests.plugins.display import PYTEST_HEADER_MODULES, TESTED_VERSIONS


packagename = os.path.basename(os.path.dirname(__file__))
TESTED_VERSIONS[packagename] = version.version

# Treat all DeprecationWarnings as exceptions
from astropy.tests.helper import enable_deprecations_as_exceptions

# TODO: add numpy again once https://github.com/astropy/regions/pull/252 is addressed
enable_deprecations_as_exceptions(warnings_to_ignore_entire_module=["numpy", "astropy"])

# Declare for which packages version numbers should be displayed
# when running the tests
PYTEST_HEADER_MODULES["cython"] = "cython"
PYTEST_HEADER_MODULES["uncertainties"] = "uncertainties"
PYTEST_HEADER_MODULES["iminuit"] = "iminuit"
PYTEST_HEADER_MODULES["astropy"] = "astropy"
PYTEST_HEADER_MODULES["regions"] = "regions"
PYTEST_HEADER_MODULES["healpy"] = "healpy"
PYTEST_HEADER_MODULES["sherpa"] = "sherpa"
PYTEST_HEADER_MODULES["gammapy"] = "gammapy"
PYTEST_HEADER_MODULES["naima"] = "naima"
PYTEST_HEADER_MODULES["reproject"] = "reproject"


def pytest_configure(config):
    """Print some info ..."""
    from .utils.testing import has_data

    print("")
    print("Gammapy test data availability:")

    has_it = "yes" if has_data("gammapy-data") else "no"
    print("gammapy-data ... {}".format(has_it))

    print("Gammapy environment variables:")

    var = os.environ.get("GAMMAPY_DATA", "not set")
    print("GAMMAPY_DATA = {}".format(var))

    try:
        # Switch to non-interactive plotting backend to avoid GUI windows
        # popping up while running the tests.
        import matplotlib

        matplotlib.use("agg")
        print('Setting matplotlib backend to "agg" for the tests.')
    except ImportError:
        pass
