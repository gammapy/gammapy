# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from numpy.testing import assert_allclose
from astropy.units import Quantity
from gammapy.astro.source import SNR, SNRTrueloveMcKee

t = Quantity([0, 1, 10, 100, 1000, 10000], "yr")
snr = SNR()
snr_mckee = SNRTrueloveMcKee()


def test_SNR_luminosity_tev():
    """Test SNR luminosity"""
    reference = [0, 0, 0, 0, 1.076e33, 1.076e33]
    assert_allclose(snr.luminosity_tev(t).value, reference, rtol=1e-3)


def test_SNR_radius():
    """Test SNR radius"""
    reference = [0, 3.085e16, 3.085e17, 3.085e18, 1.174e19, 2.950e19]
    assert_allclose(snr.radius(t).value, reference, rtol=1e-3)


def test_SNR_radius_inner():
    """Test SNR radius"""
    reference = (1 - 0.0914) * np.array(
        [0, 3.085e16, 3.085e17, 3.085e18, 1.174e19, 2.950e19]
    )
    assert_allclose(snr.radius_inner(t).value, reference, rtol=1e-3)


def test_SNRTrueloveMcKee_luminosity_tev():
    """Test SNR Truelov McKee luminosity"""
    reference = [0, 0, 0, 0, 1.076e33, 1.076e33]
    assert_allclose(snr_mckee.luminosity_tev(t).value, reference, rtol=1e-3)


def test_SNRTrueloveMcKee_radius():
    """Test SNR RTruelove McKee radius"""
    reference = [0, 1.953e17, 9.066e17, 4.208e18, 1.579e19, 4.117e19]
    assert_allclose(snr_mckee.radius(t).value, reference, rtol=1e-3)
