# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.units import Quantity
from astropy.tests.helper import assert_quantity_allclose
from ...utils.testing import requires_dependency
#import matplotlib.pyplot as plt
from ..lightcurve import LightCurve
from numpy.testing import assert_allclose


def test_lightcurve():
    lc = LightCurve.simulate_example()
    flux_mean = lc['FLUX'].mean()
    assert_quantity_allclose(flux_mean, Quantity(5.25, 'cm^-2 s^-1'))


def test_lightcurve_fvar():
    lc = LightCurve.simulate_example()
    fvar, fvarerr = lc.compute_fvar()
    assert_allclose(fvar, 0.6565905201197404)
    assert_allclose(fvarerr, 0.057795285237677206)


@requires_dependency('matplotlib')
def test_lightcurve_plot():
    lc = LightCurve.simulate_example()
    #lc = make_example_lightcurve()
    lc.plot()
