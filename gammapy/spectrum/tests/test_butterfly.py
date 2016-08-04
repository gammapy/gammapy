# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.units import Quantity
from astropy.tests.helper import assert_quantity_allclose, pytest
from ...utils.testing import requires_dependency
from ..butterfly import SpectrumButterfly
from ..crab import crab_flux


@pytest.fixture
def butterfly():
    bf = SpectrumButterfly()

    bf['energy'] = Quantity([1, 2, 10], 'TeV')
    bf['flux'] = crab_flux(bf['energy'])
    bf['flux_lo'] = [0.9, 0.8, 0.9] * bf['flux']
    bf['flux_hi'] = [1.1, 1.2, 1.1] * bf['flux']

    return bf


def test_butterfly_basics(butterfly):
    flux_mean = butterfly['flux'].mean()
    assert_quantity_allclose(flux_mean, Quantity(5.25, 'cm^-2 s^-1'))


@requires_dependency('matplotlib')
def test_butterfly_plot(butterfly):
    ax = butterfly.plot()
    # TODO: get something from `ax` and assert on it
