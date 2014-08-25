# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import unittest
from astropy.tests.helper import pytest
import numpy as np
from numpy.testing import assert_almost_equal
from ...morphology import (Gauss2DPDF,
                           MultiGauss2D,
                           ThetaCalculator,
                           ThetaCalculatorScipy,
                           )

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@pytest.mark.skipif('not HAS_SCIPY')
class TestThetaCalculator(unittest.TestCase):
    """We use a Gaussian, because it has known analytical
    solutions for theta and containment."""
    def setUp(self):
        # Single Gauss
        self.g = Gauss2DPDF(sigma=1)
        self.g_tc = ThetaCalculator(self.g.dpdtheta2, theta_max=5, n_bins=1e6)
        self.g_tcs = ThetaCalculatorScipy(self.g.dpdtheta2, theta_max=5)
        # Multi Gauss
        self.m = MultiGauss2D(sigmas=[1, 2])
        self.m_tc = ThetaCalculator(self.m.dpdtheta2, theta_max=5, n_bins=1e6)
        self.m_tcs = ThetaCalculatorScipy(self.m.dpdtheta2, theta_max=5)
        # self.tc2 = mt.ThetaCalculator2D.from_source(self.g, theta_max=5, d)

    def test_containment_Gauss2D(self):
        for tc in [self.g_tc, self.g_tcs]:
            for theta in np.linspace(0, 3, 4):
                actual = tc.containment_fraction(theta)
                desired = self.g.containment_fraction(theta)
                assert_almost_equal(actual, desired, decimal=4)

    def test_containment_MultiGauss2D(self):
        for tc in [self.m_tc, self.m_tcs]:
            for theta in np.linspace(0, 3, 4):
                actual = tc.containment_fraction(theta)
                desired = self.m.containment_fraction(theta)
                assert_almost_equal(actual, desired, decimal=4)

    def test_theta_Gauss2D(self):
        for tc in [self.g_tc, self.g_tcs]:
            for containment in np.arange(0, 1, 0.1):
                actual = tc.containment_radius(containment)
                desired = self.g.containment_radius(containment)
                assert_almost_equal(actual, desired, decimal=4)

    def test_theta_MultiGauss2D(self):
        for tc in [self.m_tc, self.m_tcs]:
            for containment in np.arange(0, 1, 0.1):
                actual = tc.containment_radius(containment)
                desired = self.m.containment_radius(containment)
                assert_almost_equal(actual, desired, decimal=4)


# FIXME: This test is slow and fails with an IndexError.
def _test_ModelThetaCalculator():
    """Check that Gaussian widths add in quadrature
    i.e. sigma_psf = 3, sigma_source = 4 ===> sigma_model = 5"""
    source, psf = Gauss2DPDF(3), Gauss2DPDF(4)
    # Correct analytical reference
    ana = Gauss2DPDF(5)
    ana_angle = ana.containment_radius(0.5)
    ana_containment = ana.containment(ana_angle)
    # Numerical method
    fov, binsz = 20, 0.2
    num = ModelThetaCalculator(source, psf, fov, binsz)
    num_angle = num.containment_radius(0.5)
    num_containment = num.containment(num_angle)
    # Compare results
    par_names = ['angle', 'containment']
    par_refs = [ana_angle, ana_containment]
    par_checks = [num_angle, num_containment]

    # TODO: add asserts
