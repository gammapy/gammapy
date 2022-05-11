# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from numpy.testing import assert_allclose
from astropy import units as u
from gammapy.utils.gauss import Gauss2DPDF, MultiGauss2D


class TestGauss2DPDF:
    """Note that we test __call__ and dpdtheta2 by
    checking that their integrals as advertised are 1."""

    def setup(self):
        self.gs = [
            Gauss2DPDF(0.1 * u.deg),
            Gauss2DPDF(1 * u.deg),
            Gauss2DPDF(1 * u.deg),
        ]

    def test_call(self):
        # Check that value at origin matches the one given here:
        # http://en.wikipedia.org/wiki/Multivariate_normal_distribution#Bivariate_case
        for g in self.gs:
            actual = g(0 * u.deg, 0 * u.deg)
            desired = 1 / (2 * np.pi * g.sigma**2)
            assert_allclose(actual, desired)

    def test_containment(self):
        for g in self.gs:
            assert_allclose(g.containment_fraction(g.sigma), 0.39346934028736658)
            assert_allclose(g.containment_fraction(2 * g.sigma), 0.8646647167633873)

    def test_theta(self):
        for g in self.gs:
            assert_allclose(g.containment_radius(0.68) / g.sigma, 1.5095921854516636)
            assert_allclose(g.containment_radius(0.95) / g.sigma, 2.4477468306808161)

    def test_gauss_convolve(self):
        g = Gauss2DPDF(sigma=3 * u.deg).gauss_convolve(sigma=4 * u.deg)
        assert_allclose(g.sigma, 5 * u.deg)


class TestMultiGauss2D:
    """Note that we test __call__ and dpdtheta2 by
    checking that their integrals."""

    @staticmethod
    def test_integral_normalize():
        m = MultiGauss2D(sigmas=[1, 2], norms=[3, 4])
        assert_allclose(m.integral, 7)
        m.normalize()
        assert_allclose(m.integral, 1)

    @staticmethod
    def test_containment():
        g, g2 = Gauss2DPDF(sigma=1), Gauss2DPDF(sigma=2)
        m = MultiGauss2D(sigmas=[1])
        m2 = MultiGauss2D(sigmas=[1, 2], norms=[3, 4])
        for theta in [0, 0.1, 1, 5]:
            assert_allclose(
                m.containment_fraction(theta), g.containment_fraction(theta)
            )
            actual = m2.containment_fraction(theta)
            desired = 3 * g.containment_fraction(theta) + 4 * g2.containment_fraction(
                theta
            )
            assert_allclose(actual, desired)

    @staticmethod
    def test_theta():
        # Closure test
        m = MultiGauss2D(sigmas=[1, 2] * u.deg, norms=[3, 4])
        for theta in [0, 0.1, 1, 5] * u.deg:
            c = m.containment_fraction(theta)
            t = m.containment_radius(c)
            assert_allclose(t, theta)

    @staticmethod
    def test_gauss_convolve():
        # Convolution must add sigmas in square
        m = MultiGauss2D(sigmas=[3], norms=[5])
        m2 = m.gauss_convolve(4, 6)
        assert_allclose(m2.sigmas, [5])
        assert_allclose(m2.integral, 5 * 6)
        # Check that convolve did not change the original
        assert_allclose(m.sigmas, [3])
        assert_allclose(m.norms, [5])
