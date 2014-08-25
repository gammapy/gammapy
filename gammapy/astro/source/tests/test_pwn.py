# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from numpy.testing import assert_allclose
import numpy as np
from astropy.units import Quantity
from astropy.tests.helper import pytest
from ...source import PWN

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


t = Quantity([0, 1, 10, 100, 1000, 10000, 100000], 'yr')
pwn = PWN()


@pytest.mark.skipif('not HAS_SCIPY')
def test_PWN_radius():
    """Test SNR luminosity"""
    reference = [0.00000000e+00, 1.33404629e+14, 2.11432089e+15,
                 3.35097278e+16, 5.31093395e+17, 6.92792702e+17, 6.92792702e+17]
    assert_allclose(pwn.radius(t), reference)


@pytest.mark.skipif('not HAS_SCIPY')
def test_magnetic_field():
    """Test SNR luminosity"""
    reference = [np.nan, 1.75348134e-03, 8.78822460e-05, 4.40454585e-06,
                 2.20750154e-07, 4.68544888e-07, 1.48162794e-06]
    assert_allclose(pwn.magnetic_field(t).to('gauss'), reference)
