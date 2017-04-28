# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities for testing"""
from __future__ import absolute_import, division, print_function, unicode_literals
import os
from astropy.coordinates import Angle
from astropy.tests.helper import pytest
from numpy.testing import assert_array_less, assert_allclose
from ..data import DataManager
from ..datasets import gammapy_extra

__all__ = [
    'requires_dependency',
    'requires_data',
    'assert_wcs_allclose',
    'assert_skycoord_allclose',
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


def run_cli(cli, args, assert_ok=True):
    """Helper function to run command line tools.
    """
    with pytest.raises(SystemExit) as exc:
        cli(args)

    if assert_ok:
        assert exc.value.args[0] == 0

    return exc


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


def assert_skycoord_allclose(skycoord1, skycoord2, atol='1 arcsec'):
    """Assert all-close for `~astropy.coordinates.SkyCoord`.

    - Checks if separation on the sky is within ``atol``.
    - Frames can be different, aren't checked at the moment.
    """
    atol = Angle(atol)
    sep = skycoord1.separation(skycoord2).deg
    assert_array_less(sep.deg, atol.deg)
