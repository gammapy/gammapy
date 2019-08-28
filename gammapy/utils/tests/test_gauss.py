# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import scipy.integrate
from numpy.testing import assert_almost_equal, assert_equal
from gammapy.utils.gauss import Gauss2DPDF, MultiGauss2D


class TestGauss2DPDF:
    """Note that we test __call__ and dpdtheta2 by
    checking that their integrals as advertised are 1."""

    def setup(self):
        self.gs = [Gauss2DPDF(0.1), Gauss2DPDF(1), Gauss2DPDF(1)]

    def test_call(self):
        # Check that value at origin matches the one given here:
        # http://en.wikipedia.org/wiki/Multivariate_normal_distribution#Bivariate_case
        for g in self.gs:
            actual = g(0, 0)
            desired = 1 / (2 * np.pi * g.sigma ** 2)
            assert_almost_equal(actual, desired)
            # Check that distribution integrates to 1
            xy_max = 5 * g.sigma  # integration range
            integral = scipy.integrate.dblquad(
                g, -xy_max, xy_max, lambda _: -xy_max, lambda _: xy_max
            )[0]
            assert_almost_equal(integral, 1, decimal=5)

    def test_dpdtheta2(self):
        for g in self.gs:
            theta2_max = (7 * g.sigma) ** 2
            integral = scipy.integrate.quad(g.dpdtheta2, 0, theta2_max)[0]
            assert_almost_equal(integral, 1, decimal=5)

    def test_containment(self):
        for g in self.gs:
            assert_almost_equal(g.containment_fraction(g.sigma), 0.39346934028736658)
            assert_almost_equal(g.containment_fraction(2 * g.sigma), 0.8646647167633873)

    def test_theta(self):
        for g in self.gs:
            assert_almost_equal(
                g.containment_radius(0.68) / g.sigma, 1.5095921854516636
            )
            assert_almost_equal(
                g.containment_radius(0.95) / g.sigma, 2.4477468306808161
            )

    def test_gauss_convolve(self):
        g = Gauss2DPDF(sigma=3).gauss_convolve(sigma=4)
        assert_equal(g.sigma, 5)


class TestMultiGauss2D:
    """Note that we test __call__ and dpdtheta2 by
    checking that their integrals."""

    @staticmethod
    def test_call():
        m = MultiGauss2D(sigmas=[1, 2], norms=[3, 4])
        xy_max = 5 * m.max_sigma  # integration range
        integral = scipy.integrate.dblquad(
            m, -xy_max, xy_max, lambda _: -xy_max, lambda _: xy_max
        )[0]
        assert_almost_equal(integral, 7, decimal=5)

    @staticmethod
    def test_dpdtheta2():
        m = MultiGauss2D(sigmas=[1, 2], norms=[3, 4])
        theta2_max = (7 * m.max_sigma) ** 2
        integral = scipy.integrate.quad(m.dpdtheta2, 0, theta2_max)[0]
        assert_almost_equal(integral, 7, decimal=5)

    @staticmethod
    def test_integral_normalize():
        m = MultiGauss2D(sigmas=[1, 2], norms=[3, 4])
        assert_equal(m.integral, 7)
        m.normalize()
        assert_equal(m.integral, 1)

    @staticmethod
    def test_containment():
        g, g2 = Gauss2DPDF(sigma=1), Gauss2DPDF(sigma=2)
        m = MultiGauss2D(sigmas=[1])
        m2 = MultiGauss2D(sigmas=[1, 2], norms=[3, 4])
        for theta in [0, 0.1, 1, 5]:
            assert_almost_equal(
                m.containment_fraction(theta), g.containment_fraction(theta)
            )
            actual = m2.containment_fraction(theta)
            desired = 3 * g.containment_fraction(theta) + 4 * g2.containment_fraction(
                theta
            )
            assert_almost_equal(actual, desired)

    @staticmethod
    def test_theta():
        # Closure test
        m = MultiGauss2D(sigmas=[1, 2], norms=[3, 4])
        for theta in [0, 0.1, 1, 5]:
            c = m.containment_fraction(theta)
            t = m.containment_radius(c)
            assert_almost_equal(t, theta, decimal=5)

    @staticmethod
    def test_gauss_convolve():
        # Convolution must add sigmas in square
        m = MultiGauss2D(sigmas=[3], norms=[5])
        m2 = m.gauss_convolve(4, 6)
        assert_equal(m2.sigmas, [5])
        assert_almost_equal(m2.integral, 5 * 6)
        # Check that convolve did not change the original
        assert_equal(m.sigmas, [3])
        assert_equal(m.norms, [5])
