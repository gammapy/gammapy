# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
from ...utils.testing import requires_data, data_manager, requires_dependency


@requires_data('gammapy-extra')
@requires_dependency('yaml')
def test_DataManager(data_manager):
    # TODO: add asserts on info output

    data_manager.info()

    with pytest.raises(KeyError):
        ds = data_manager['kronka-lonka']

    ds = data_manager['hess-crab4-hd-hap-prod2']

    for ds in data_manager.stores:
        ds.info()
