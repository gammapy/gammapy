# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
import numpy as np
from astropy.units import Quantity
from ....utils.testing import requires_dependency
from ...source import PWN

t = Quantity([0, 1, 10, 100, 1000, 10000, 100000], "yr")
pwn = PWN()


@requires_dependency("scipy")
def test_PWN_radius():
    """Test SNR luminosity"""
    r = [0, 1.334e+14, 2.114e+15, 3.350e+16, 5.310e+17, 6.927e+17, 6.927e+17]
    assert_allclose(pwn.radius(t).value, r, rtol=1e-3)


@requires_dependency("scipy")
def test_magnetic_field():
    """Test SNR luminosity"""
    b = [np.nan, 1.753e-03, 8.788e-05, 4.404e-06, 2.207e-07, 4.685e-07, 1.481e-06]
    assert_allclose(pwn.magnetic_field(t).to("gauss").value, b, rtol=1e-3)
