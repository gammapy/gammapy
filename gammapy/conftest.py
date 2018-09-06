# this contains imports plugins that configure py.test for astropy tests.
# by importing them here in conftest.py they are discoverable by py.test
# no matter how it is invoked within the source tree.
from astropy.tests.pytest_plugins import *
import os

# This is to figure out the affiliated package version, rather than
# using Astropy's
from . import version

packagename = os.path.basename(os.path.dirname(__file__))
TESTED_VERSIONS[packagename] = version.version

# Treat all DeprecationWarnings as exceptions
# enable_deprecations_as_exceptions()

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

    has_it = "yes" if has_data("gammapy-extra") else "no"
    print("gammapy-extra ... {}".format(has_it))

    print("Gammapy environment variables:")

    var = os.environ.get("GAMMAPY_EXTRA", "not set")
    print("GAMMAPY_EXTRA = {}".format(var))

    try:
        # Switch to non-interactive plotting backend to avoid GUI windows
        # popping up while running the tests.
        import matplotlib

        matplotlib.use("agg")
        print('Setting matplotlib backend to "agg" for the tests.')
    except ImportError:
        pass
