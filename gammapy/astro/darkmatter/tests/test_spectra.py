# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import astropy.units as u
from ....utils.testing import assert_quantity_allclose, requires_data
from .. import PrimaryFlux


@requires_data("gammapy-extra")
def test_primary_flux():
    with pytest.raises(ValueError):
        PrimaryFlux(channel="Spam", mDM=1 * u.TeV)

    primflux = PrimaryFlux(channel="W", mDM=1 * u.TeV)
    actual = primflux.table_model(500 * u.GeV)
    desired = 9.328234e-05 / u.GeV
    assert_quantity_allclose(actual, desired)
