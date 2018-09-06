# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
import numpy as np
from astropy.units import Quantity
from ...source import SNR, SNRTrueloveMcKee

t = Quantity([0, 1, 10, 100, 1000, 10000], "yr")
snr = SNR()
snr_mckee = SNRTrueloveMcKee()


def test_SNR_luminosity_tev():
    """Test SNR luminosity"""
    reference = [0, 0, 0, 0, 1.076e+33, 1.076e+33]
    assert_allclose(snr.luminosity_tev(t).value, reference, rtol=1e-3)


def test_SNR_radius():
    """Test SNR radius"""
    reference = [0, 3.085e+16, 3.085e+17, 3.085e+18, 1.174e+19, 2.950e+19]
    assert_allclose(snr.radius(t).value, reference, rtol=1e-3)


def test_SNR_radius_inner():
    """Test SNR radius"""
    reference = (1 - 0.0914) * np.array(
        [0, 3.085e+16, 3.085e+17, 3.085e+18, 1.174e+19, 2.950e+19]
    )
    assert_allclose(snr.radius_inner(t).value, reference, rtol=1e-3)


def test_SNRTrueloveMcKee_luminosity_tev():
    """Test SNR Truelov McKee luminosity"""
    reference = [0, 0, 0, 0, 1.076e+33, 1.076e+33]
    assert_allclose(snr_mckee.luminosity_tev(t).value, reference, rtol=1e-3)


def test_SNRTrueloveMcKee_radius():
    """Test SNR RTruelove McKee radius"""
    reference = [0, 1.953e+17, 9.066e+17, 4.208e+18, 1.579e+19, 4.117e+19]
    assert_allclose(snr_mckee.radius(t).value, reference, rtol=1e-3)
