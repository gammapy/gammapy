# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import division
from numpy.testing import assert_approx_equal
from ..snr import SNR, SNR_Truelove


def test_SNR():
    snr = SNR()
    assert_approx_equal(snr.L(1e3), 1.0768456645602824e+33)


def test_SNR_Truelove():
    snr = SNR_Truelove()
    assert_approx_equal(snr.L(1e3), 1.0768456645602824e+33)
    assert_approx_equal(snr.r_out(1e3), 5.1177107050629278)
