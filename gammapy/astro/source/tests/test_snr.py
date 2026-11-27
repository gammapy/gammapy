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


def test_SNR_default_parameters():
    """Test SNR with default parameters (None values).

    This test covers the change from mutable default arguments to None
    in the SNR.__init__ method (B008 ruff rule fix).
    """
    # Test with all defaults
    snr_default = SNR()
    assert snr_default.theta.value == 0.1
    assert snr_default.n_ISM.value == 1
    assert snr_default.t_stop.value == 1e6

    # Test with explicit None values
    snr_none = SNR(theta=None, n_ISM=None, t_stop=None)
    assert snr_none.theta.value == 0.1
    assert snr_none.n_ISM.value == 1
    assert snr_none.t_stop.value == 1e6

    # Test with custom values
    snr_custom = SNR(
        theta=Quantity(0.2), n_ISM=Quantity(2, "cm-3"), t_stop=Quantity(2e6, "K")
    )
    assert snr_custom.theta.value == 0.2
    assert snr_custom.n_ISM.value == 2
    assert snr_custom.t_stop.value == 2e6


def test_SNRTrueloveMcKee_default_parameters():
    """Test SNRTrueloveMcKee with default parameters (None values).

    This test covers the change from mutable default arguments to None
    in the SNR.__init__ method which SNRTrueloveMcKee inherits from.
    """
    # Test with all defaults
    snr_mckee_default = SNRTrueloveMcKee()
    assert snr_mckee_default.theta.value == 0.1
    assert snr_mckee_default.n_ISM.value == 1
    assert snr_mckee_default.t_stop.value == 1e6
