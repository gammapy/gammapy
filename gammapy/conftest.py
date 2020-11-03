# Licensed under a 3-clause BSD style license - see LICENSE.rst
# this contains imports plugins that configure py.test for astropy tests.
# by importing them here in conftest.py they are discoverable by py.test
# no matter how it is invoked within the source tree.
import os
from astropy.tests.helper import enable_deprecations_as_exceptions
from pytest_astropy_header.display import PYTEST_HEADER_MODULES

# TODO: activate this again and handle deprecations in the code
# enable_deprecations_as_exceptions(warnings_to_ignore_entire_module=["numpy", "astropy"])

# Declare for which packages version numbers should be displayed
# when running the tests
PYTEST_HEADER_MODULES["cython"] = "cython"
PYTEST_HEADER_MODULES["iminuit"] = "iminuit"
PYTEST_HEADER_MODULES["astropy"] = "astropy"
PYTEST_HEADER_MODULES["regions"] = "regions"
PYTEST_HEADER_MODULES["healpy"] = "healpy"
PYTEST_HEADER_MODULES["sherpa"] = "sherpa"
PYTEST_HEADER_MODULES["gammapy"] = "gammapy"
PYTEST_HEADER_MODULES["naima"] = "naima"


def pytest_configure(config):
    """Print some info ..."""
    from gammapy.utils.testing import has_data

    config.option.astropy_header = True

    print("")
    print("Gammapy test data availability:")

    has_it = "yes" if has_data("gammapy-data") else "no"
    print(f"gammapy-data ... {has_it}")

    print("Gammapy environment variables:")

    var = os.environ.get("GAMMAPY_DATA", "not set")
    print(f"GAMMAPY_DATA = {var}")

    try:
        # Switch to non-interactive plotting backend to avoid GUI windows
        # popping up while running the tests.
        import matplotlib

        matplotlib.use("agg")
        print('Setting matplotlib backend to "agg" for the tests.')
    except ImportError:
        pass
