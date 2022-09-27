# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from numpy.testing import assert_allclose
from astropy.units import Quantity
from gammapy.astro.source import PWN

t = Quantity([0, 1, 10, 100, 1000, 10000, 100000], "yr")
pwn = PWN()


def test_PWN_radius():
    """Test SNR luminosity"""
    r = [0, 1.334e14, 2.114e15, 3.350e16, 5.310e17, 6.927e17, 6.927e17]
    assert_allclose(pwn.radius(t).to_value("cm"), r, rtol=1e-3)


def test_magnetic_field():
    """Test SNR luminosity"""
    b = [np.nan, 1.753e-03, 8.788e-05, 4.404e-06, 2.207e-07, 4.685e-07, 1.481e-06]
    assert_allclose(pwn.magnetic_field(t).to_value("gauss"), b, rtol=1e-3)
