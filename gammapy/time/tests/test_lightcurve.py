# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.units import Quantity
from astropy.tests.helper import assert_quantity_allclose
from ...utils.testing import requires_dependency
from ..lightcurve import LightCurve
from numpy.testing import assert_allclose


def test_lightcurve():
    lc = LightCurve.simulate_example()
    flux_mean = lc['FLUX'].mean()
    assert_quantity_allclose(flux_mean, Quantity(5.25, 'cm^-2 s^-1'))


def test_lightcurve_fvar():
    lc = LightCurve.simulate_example()
    fvar, fvar_err = lc.compute_fvar()
    assert_allclose(fvar, 0.6565905201197404)
    # Note: the following tolerance is very low in the next assert,
    # because results differ by ~ 1e-3 :
    # travis-ci result: 0.05773502691896258
    # Christoph's Macbook: 0.057795285237677206
    assert_allclose(fvar_err, 0.057795285237677206, rtol=1e-2)


@requires_dependency('matplotlib')
def test_lightcurve_plot():
    lc = LightCurve.simulate_example()
    lc.plot()
