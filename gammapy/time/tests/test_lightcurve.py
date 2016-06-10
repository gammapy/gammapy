# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_almost_equal
from astropy.units import Quantity
from astropy.tests.helper import assert_quantity_allclose
from ..lightcurve import LightCurve, make_example_lightcurve


def test_lightcurve():
    lc = make_example_lightcurve()	
    flux_mean = lc['FLUX'].mean() 
    assert_quantity_allclose(flux_mean, Quantity(5.25,'cm^-2 s^-1'))

def test_lightcurve_plot():
    lc = make_example_lightcurve()	
    lc.lc_plot()
