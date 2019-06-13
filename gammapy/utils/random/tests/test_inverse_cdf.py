# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import scipy
import scipy.stats as stats
from numpy.testing import assert_allclose

from ..inverse_cdf import InverseCDFSampler

def uniform_dist(n_elem):
    return np.ones(n_elem) / n_elem

def gauss_dist(x, mu=0, sigma=0.1):
    return stats.norm.pdf(x,mu,sigma)

def test_uniform_dist_sampling():
    n_elem = 100000
    n_sampled = 1000
    bins = 100
    idx = InverseCDFSampler(uniform_dist(n_elem=n_elem), random_state=0)
    sampled_elem = idx.sample(n_sampled)
    hist = np.histogram(sampled_elem[0],bins=bins)
    assert_allclose((np.mean(hist[0]) * 100./1000.), 1.0, atol=1e-4)

def test_norm_dist_sampling():
    n_sampled = 1000000
    x = np.linspace(-10, 10, 1000)
    sigma = 0.01
    sampler = InverseCDFSampler(gauss_dist(x=x, mu=0, sigma=sigma), random_state=0)
    idx = sampler.sample(int(n_sampled))
    x_sampled = np.interp(idx, np.arange(1000), x)
    assert_allclose(np.mean(x_sampled), 0.0, atol=0.01)
    assert_allclose(np.std(x_sampled), 0.01, atol=0.005)

