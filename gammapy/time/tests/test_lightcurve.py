# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.units import Quantity
from astropy.tests.helper import assert_quantity_allclose
from ...utils.testing import requires_dependency
from ..lightcurve import LightCurve


def test_lightcurve():
    lc = LightCurve.simulate_example()
    flux_mean = lc['FLUX'].mean()
    assert_quantity_allclose(flux_mean, Quantity(5.25, 'cm^-2 s^-1'))


@requires_dependency('matplotlib')
def test_lightcurve_plot():
    lc = LightCurve.simulate_example()
    lc.plot()
