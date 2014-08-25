# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import unittest
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal, assert_allclose
from astropy.tests.helper import pytest
from astropy.modeling.models import Gaussian2D
from astropy.convolution import discretize_model
from ...image import measure_image_moments
from ...morphology import Gauss2DPDF, MultiGauss2D, gaussian_sum_moments

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import uncertainties
    HAS_UNCERTAINTIES = True
except ImportError:
    HAS_UNCERTAINTIES = False


@pytest.mark.skipif('not HAS_SCIPY')
class TestGauss2DPDF(unittest.TestCase):
    """Note that we test __call__ and dpdtheta2 by
    checking that their integrals as advertised are 1."""
    def setUp(self):
        self.gs = [Gauss2DPDF(0.1), Gauss2DPDF(1), Gauss2DPDF(1)]

    def test_call(self):
        from scipy.integrate import dblquad
        # Check that value at origin matches the one given here:
        # http://en.wikipedia.org/wiki/Multivariate_normal_distribution#Bivariate_case
        for g in self.gs:
            actual = g(0, 0)
            desired = 1 / (2 * np.pi * g.sigma ** 2)
            assert_almost_equal(actual, desired)
            # Check that distribution integrates to 1
            xy_max = 5 * g.sigma  # integration range
            integral = dblquad(g, -xy_max, xy_max,
                               lambda _: -xy_max, lambda _: xy_max)[0]
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
            assert_almost_equal(g.containment_radius(0.68) / g.sigma, 1.5095921854516636)
            assert_almost_equal(g.containment_radius(0.95) / g.sigma, 2.4477468306808161)

    def test_gauss_convolve(self):
        g = Gauss2DPDF(sigma=3).gauss_convolve(sigma=4)
        assert_equal(g.sigma, 5)


@pytest.mark.skipif('not HAS_SCIPY')
class TestMultiGauss2D(unittest.TestCase):
    """Note that we test __call__ and dpdtheta2 by
    checking that their integrals."""
    def test_call(self):
        from scipy.integrate import dblquad
        m = MultiGauss2D(sigmas=[1, 2], norms=[3, 4])
        xy_max = 5 * m.max_sigma  # integration range
        integral = dblquad(m, -xy_max, xy_max,
                           lambda _: -xy_max, lambda _: xy_max)[0]
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
        g, g2 = Gauss2DPDF(sigma=1), Gauss2DPDF(sigma=2)
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

    def test_gauss_convolve(self):
        # Convolution must add sigmas in square
        m = MultiGauss2D(sigmas=[3], norms=[5])
        m2 = m.gauss_convolve(4, 6)
        assert_equal(m2.sigmas, [5])
        assert_almost_equal(m2.integral, 5 * 6)
        # Check that convolve did not change the original
        assert_equal(m.sigmas, [3])
        assert_equal(m.norms, [5])


@pytest.mark.skipif('not HAS_UNCERTAINTIES')
def test_gaussian_sum_moments():
    """Check analytical against numerical solution.
    """

    # We define three components with different flux, position and size
    F_1, F_2, F_3 = 100, 200, 300
    sigma_1, sigma_2, sigma_3 = 15, 10, 5
    x_1, x_2, x_3 = 100, 120, 70
    y_1, y_2, y_3 = 100, 90, 120

    # Convert into non-normalized amplitude for astropy model
    def A(F, sigma):
        return F * 1 / (2 * np.pi * sigma ** 2)

    # Define and evaluate models
    f_1 = Gaussian2D(A(F_1, sigma_1), x_1, y_1, sigma_1, sigma_1)
    f_2 = Gaussian2D(A(F_2, sigma_2), x_2, y_2, sigma_2, sigma_2)
    f_3 = Gaussian2D(A(F_3, sigma_3), x_3, y_3, sigma_3, sigma_3)

    F_1_image = discretize_model(f_1, (0, 200), (0, 200))
    F_2_image = discretize_model(f_2, (0, 200), (0, 200))
    F_3_image = discretize_model(f_3, (0, 200), (0, 200))

    moments_num = measure_image_moments(F_1_image + F_2_image + F_3_image)

    # Compute analytical values
    cov_matrix = np.zeros((12, 12))
    F = [F_1, F_2, F_3]
    sigma = [sigma_1, sigma_2, sigma_3]
    x = [x_1, x_2, x_3]
    y = [y_1, y_2, y_3]

    moments_ana, uncertainties = gaussian_sum_moments(F, sigma, x, y, cov_matrix)
    assert_allclose(moments_ana, moments_num, 1e-6)
    assert_allclose(uncertainties, 0)
