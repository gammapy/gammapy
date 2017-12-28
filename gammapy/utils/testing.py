# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities for testing"""
from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import os
import pytest
from astropy.time import Time
from astropy.coordinates import SkyCoord
from numpy.testing import assert_allclose
from ..data import DataManager
from ..datasets import gammapy_extra

__all__ = [
    'requires_dependency',
    'requires_data',
    'assert_wcs_allclose',
    'assert_skycoord_allclose',
    'assert_time_allclose',
]

# Cache for `requires_dependency`
_requires_dependency_cache = dict()


def requires_dependency(name):
    """Decorator to declare required dependencies for tests.

    Examples
    --------

    ::

        from gammapy.utils.testing import requires_dependency

        @requires_dependency('scipy')
        def test_using_scipy():
            import scipy
            ...
    """
    if name in _requires_dependency_cache:
        skip_it = _requires_dependency_cache[name]
    else:
        try:
            __import__(name)
            skip_it = False
        except ImportError:
            skip_it = True

        _requires_dependency_cache[name] = skip_it

    reason = 'Missing dependency: {}'.format(name)
    return pytest.mark.skipif(skip_it, reason=reason)


def has_hess_test_data():
    """Check if the user has HESS data for testing.

    """
    if not DataManager.DEFAULT_CONFIG_FILE.is_file():
        return False

    try:
        dm = DataManager()
        # TODO: add checks on availability of datasets used in the tests ...
        return True
    except:
        return False


def has_data(name):
    """Is a certain set of data available?
    """
    if name == 'gammapy-extra':
        from ..datasets import gammapy_extra
        return gammapy_extra.is_available
    elif name == 'hess':
        return has_hess_test_data()
    elif name == 'hgps':
        return ('HGPS_DATA' in os.environ) and ('HGPS_ANALYSIS' in os.environ)
    elif name == 'gamma-cat':
        return ('GAMMA_CAT' in os.environ)
    elif name == 'fermi-lat':
        return ('GAMMAPY_FERMI_LAT_DATA' in os.environ)
    else:
        raise ValueError('Invalid name: {}'.format(name))


def requires_data(name):
    """Decorator to declare required data for tests.

    Examples
    --------

    ::

        from gammapy.utils.testing import requires_data
        from gammapy.datasets import gammapy_extra

        @requires_data('gammapy-extra')
        def test_using_data_files():
            filename = gammapy_extra.filename('...')
            ...
    """
    skip_it = not has_data(name)

    reason = 'Missing data: {}'.format(name)
    return pytest.mark.skipif(skip_it, reason=reason)


def run_cli(cli, args, exit_code=0):
    """Run Click command line tool.

    Thin wrapper around `click.testing.CliRunner`
    that prints info to stderr if the command fails.

    Parameters
    ----------
    cli : click.Command
        Click command
    args : list of str
        Argument list
    exit_code : int
        Expected exit code of the command

    Returns
    -------
    result : `click.testing.Result`
        Result
    """
    from click.testing import CliRunner
    result = CliRunner().invoke(cli, args, catch_exceptions=False)

    if result.exit_code != exit_code:
        sys.stderr.write('Exit code mismatch!\n')
        sys.stderr.write('Ouput:\n')
        sys.stderr.write(result.output)

    return result


# https://pytest.org/latest/tmpdir.html#the-tmpdir-factory-fixture
@pytest.fixture
def data_manager():
    test_register = gammapy_extra.filename('datasets/data-register.yaml')
    dm = DataManager.from_yaml(test_register)
    return dm


def assert_wcs_allclose(wcs1, wcs2):
    """Assert all-close for `~astropy.wcs.WCS`

    """
    # TODO: implement properly
    assert_allclose(wcs1.wcs.cdelt, wcs2.wcs.cdelt)


def assert_skycoord_allclose(actual, desired):
    """Assert all-close for `~astropy.coordinates.SkyCoord`.

    - Frames can be different, aren't checked at the moment.
    """
    assert isinstance(actual, SkyCoord)
    assert isinstance(desired, SkyCoord)
    assert_allclose(actual.data.lon.value, desired.data.lon.value)
    assert_allclose(actual.data.lat.value, desired.data.lat.value)


def assert_time_allclose(actual, desired):
    """Assert that two `astropy.time.Time` objects are almost the same."""
    assert isinstance(actual, Time)
    assert isinstance(desired, Time)
    assert_allclose(actual.value, desired.value)
    assert actual.scale == desired.scale
    assert actual.format == desired.format


def mpl_savefig_check():
    """Call matplotlib savefig for the current figure.

    This will trigger a render of the Figure, which can sometimes
    raise errors if there is a problem; i.e. we call this at the
    end of every plotting test for now.

    This is writing to an in-memory byte buffer, i.e. is faster
    than writing to disk.
    """
    import matplotlib.pyplot as plt
    from io import BytesIO
    plt.savefig(BytesIO(), format='png')
