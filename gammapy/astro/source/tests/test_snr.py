# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
import numpy as np
from astropy.units import Quantity
from ...source import SNR, SNRTrueloveMcKee

t = Quantity([0, 1, 10, 100, 1000, 10000], 'yr')
snr = SNR()
snr_mckee = SNRTrueloveMcKee()


def test_SNR_luminosity_tev():
    """Test SNR luminosity"""
    reference = [
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        1.07680000e+33,
        1.07680000e+33,
    ]
    assert_allclose(snr.luminosity_tev(t).value, reference)


def test_SNR_radius():
    """Test SNR radius"""
    reference = [
        0.00000000e+00,
        3.08567758e+16,
        3.08567758e+17,
        3.08567758e+18,
        1.17481246e+19,
        2.95099547e+19,
    ]
    assert_allclose(snr.radius(t).value, reference)


def test_SNR_radius_inner():
    """Test SNR radius"""
    reference = np.array(
        [
            0.00000000e+00,
            3.08567758e+16,
            3.08567758e+17,
            3.08567758e+18,
            1.17481246e+19,
            2.95099547e+19,
        ]
    )
    assert_allclose(snr.radius_inner(t).value, reference * (1 - 0.0914))


def test_SNRTrueloveMcKee_luminosity_tev():
    """Test SNR Truelov McKee luminosity"""
    reference = [
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        0.00000000e+00,
        1.07680000e+33,
        1.07680000e+33,
    ]
    assert_allclose(snr_mckee.luminosity_tev(t).value, reference)


def test_SNRTrueloveMcKee_radius():
    """Test SNR RTruelove McKee radius"""
    reference = [
        0.00000000e+00,
        1.95327725e+17,
        9.06630987e+17,
        4.20820826e+18,
        1.57916052e+19,
        4.11702961e+19,
    ]
    assert_allclose(snr_mckee.radius(t).value, reference, rtol=1e-3)
