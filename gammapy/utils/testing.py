# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utilities for testing"""
from __future__ import absolute_import, division, print_function, unicode_literals
import os
from astropy.tests.helper import pytest
from astropy.utils.data import get_pkg_data_filename
from ..data import DataManager

__all__ = [
    'requires_dependency',
    'requires_data',
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


# https://pytest.org/latest/tmpdir.html#the-tmpdir-factory-fixture
@pytest.fixture
def data_manager():
    test_register = gammapy_extra.filename('test_datasets/test-data-register.yaml')
    return DataManager.from_yaml(test_register)
    

