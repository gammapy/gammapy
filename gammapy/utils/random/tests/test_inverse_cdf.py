# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import scipy.stats as stats
from numpy.testing import assert_allclose
from gammapy.utils.random import InverseCDFSampler


def uniform_dist(x, a, b):
    return np.select([x <= a, x >= b], [0, 0], 1 / (b - a))


def gauss_dist(x, mu, sigma):
    return stats.norm.pdf(x, mu, sigma)


def test_uniform_dist_sampling():
    n_sampled = 1000
    x = np.linspace(-2, 2, n_sampled)

    a, b = -1, 1
    pdf = uniform_dist(x, a=a, b=b)
    sampler = InverseCDFSampler(pdf=pdf, random_state=0)

    idx = sampler.sample(int(1e4))
    x_sampled = np.interp(idx, np.arange(n_sampled), x)

    assert_allclose(np.mean(x_sampled), 0.5 * (a + b), atol=0.01)
    assert_allclose(
        np.std(x_sampled), np.sqrt(1 / 3 * (a**2 + a * b + b**2)), rtol=0.01
    )


def test_norm_dist_sampling():
    n_sampled = 1000
    x = np.linspace(-2, 2, n_sampled)

    mu, sigma = 0, 0.1
    pdf = gauss_dist(x=x, mu=mu, sigma=sigma)
    sampler = InverseCDFSampler(pdf=pdf, random_state=0)

    idx = sampler.sample(int(1e5))
    x_sampled = np.interp(idx, np.arange(n_sampled), x)

    assert_allclose(np.mean(x_sampled), mu, atol=0.01)
    assert_allclose(np.std(x_sampled), sigma, atol=0.005)


def test_axis_sampling():
    n_sampled = 1000
    x = np.linspace(-2, 2, n_sampled)

    a, b = -1, 1
    pdf_uniform = uniform_dist(x, a=a, b=b)

    mu, sigma = 0, 0.1
    pdf_gauss = gauss_dist(x=x, mu=mu, sigma=sigma)

    pdf = np.vstack([pdf_gauss, pdf_uniform])
    sampler = InverseCDFSampler(pdf, random_state=0, axis=1)

    idx = sampler.sample_axis()
    x_sampled = np.interp(idx, np.arange(n_sampled), x)

    assert_allclose(x_sampled, [0.012266, 0.43081], rtol=1e-4)
