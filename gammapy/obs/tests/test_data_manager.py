# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.tests.helper import pytest
from . import _HAS_TEST_DATA, data_manager


@pytest.mark.skipif('not _HAS_TEST_DATA')
def test_DataManager(data_manager):
    # TODO: add asserts on info output

    data_manager.info()

    with pytest.raises(KeyError):
        ds = data_manager['kronka-lonka']

    ds = data_manager['hess-paris-prod02']

    for ds in data_manager.stores:
        ds.info()
