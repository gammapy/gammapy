# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import unittest
import pytest
from numpy import pi
from numpy.testing import assert_equal, assert_almost_equal
from ..gauss import Gauss2D, MultiGauss2D

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@unittest.skip
@pytest.mark.skipif('not HAS_SCIPY')
class TestGauss2D(unittest.TestCase):
    """Note that we test __call__ and dpdtheta2 by
    checking that their integrals as advertised are 1."""
    def setUp(self):
        self.gs = [Gauss2D(0.1), Gauss2D(1), Gauss2D(1)]

    def test_call(self):
        from scipy.integrate import dblquad
        # Check that value at origin matches the one given here:
        # http://en.wikipedia.org/wiki/Multivariate_normal_distribution#Bivariate_case
        for g in self.gs:
            actual = g(0, 0)
            desired = 1 / (2 * pi * g.sigma ** 2)
            assert_almost_equal(actual, desired)
            # Check that distribution integrates to 1
            xy_max = 5 * g.sigma  # integration range
            integral = dblquad(g, -xy_max, xy_max,
                               lambda _:-xy_max, lambda _: xy_max)[0]
            assert_almost_equal(integral, 1, decimal=5)

    def test_dpdtheta2(self):
        from scipy.integrate import quad
        for g in self.gs:
            theta2_max = (7 * g.sigma) ** 2
            integral = quad(g.dpdtheta2, 0, theta2_max)[0]
            assert_almost_equal(integral, 1, decimal=5)

    def test_containment(self):
        for g in self.gs:
            assert_almost_equal(g.containment_fraction(g.sigma), 0.39346934028736658)
            assert_almost_equal(g.containment_fraction(2 * g.sigma), 0.8646647167633873)

    def test_theta(self):
        for g in self.gs:
            assert_almost_equal(g.containment_radius(0.68), g.sigma * 1.5095921854516636)
            assert_almost_equal(g.containment_radius(0.95), g.sigma * 2.4477467332775422)

    def test_convolve(self):
        g = Gauss2D(sigma=3).convolve(sigma=4)
        assert_equal(g.sigma, 5)


class TestMultiGauss2D(unittest.TestCase):
    """Note that we test __call__ and dpdtheta2 by
    checking that their integrals."""
    def test_call(self):
        from scipy.integrate import dblquad
        m = MultiGauss2D(sigmas=[1, 2], norms=[3, 4])
        xy_max = 5 * m.max_sigma  # integration range
        integral = dblquad(m, -xy_max, xy_max,
                           lambda _:-xy_max, lambda _: xy_max)[0]
        assert_almost_equal(integral, 7, decimal=5)

    def test_dpdtheta2(self):
        from scipy.integrate import quad
        m = MultiGauss2D(sigmas=[1, 2], norms=[3, 4])
        theta2_max = (7 * m.max_sigma) ** 2
        integral = quad(m.dpdtheta2, 0, theta2_max)[0]
        assert_almost_equal(integral, 7, decimal=5)

    def test_integral_normalize(self):
        m = MultiGauss2D(sigmas=[1, 2], norms=[3, 4])
        assert_equal(m.integral, 7)
        m.normalize()
        assert_equal(m.integral, 1)

    def test_containment(self):
        g, g2 = Gauss2D(sigma=1), Gauss2D(sigma=2)
        m = MultiGauss2D(sigmas=[1])
        m2 = MultiGauss2D(sigmas=[1, 2], norms=[3, 4])
        for theta in [0, 0.1, 1, 5]:
            assert_almost_equal(m.containment_fraction(theta), g.containment_fraction(theta))
            actual = m2.containment_fraction(theta)
            desired = 3 * g.containment_fraction(theta) + 4 * g2.containment_fraction(theta)
            assert_almost_equal(actual, desired)

    def test_theta(self):
        # Closure test
        m = MultiGauss2D(sigmas=[1, 2], norms=[3, 4])
        for theta in [0, 0.1, 1, 5]:
            c = m.containment_fraction(theta)
            t = m.containment_radius(c)
            assert_almost_equal(t, theta, decimal=5)

    def test_convolve(self):
        # Convolution must add sigmas in square
        m = MultiGauss2D(sigmas=[3], norms=[5])
        m2 = m.convolve(4, 6)
        assert_equal(m2.sigmas, [5])
        assert_almost_equal(m2.integral, 5 * 6)
        # Check that convolve did not change the original
        assert_equal(m.sigmas, [3])
        assert_equal(m.norms, [5])
        # Now check that convolve_me changes in place
        m.convolve_me(4, 6)
        assert_equal(m.sigmas, [5])
        assert_almost_equal(m.integral, 5 * 6)
