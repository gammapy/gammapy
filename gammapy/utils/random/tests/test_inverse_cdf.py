# Licensed under a 3-clause BSD style license - see LICENSE.rst

from ..inverse_cdf import InverseCDFSampler
import numpy as np
from numpy.testing import assert_allclose
import scipy.optimize
import scipy.stats as stats


def uniform_dist(n_elem):
    return np.ones(n_elem)

def gauss_dist(x, mu=0, sigma=0.1):
    return stats.norm.pdf(x,mu,sigma)

def gaus(x, k, sig, mu):
    return k * np.exp((-(x-mu)**2)/(2*sig**2))   # 1 gauss

def test_uniform_dist_sampling():
    for i in np.arange(1,100):
        idx = InverseCDFSampler(uniform_dist(n_elem=100000), random_state=0)
        sampled_elem = idx.sample(1000)
        hist = np.histogram(sampled_elem[0],bins=100)
        assert_allclose((np.mean(hist[0]) * 100./1000.), 1.0, atol=1e-4)

def test_norm_dist_sampling():
    for i in np.arange(1,100):
        x = np.arange(-i,i,i/1000.)
        sigma = i/100.
        idx = InverseCDFSampler(gauss_dist(x=x, mu=0, sigma=sigma), random_state=0)
        sampled_elem = idx.sample(1000)
        a1, b1 = np.histogram((sampled_elem[0]),bins=100)
        b2 = (b1 + np.roll(b1,1))/2.0
        b2 = np.delete(b2, 0)
        guess = np.array([50, 0.05, 1000])
        fitParams, fitCovariances = scipy.optimize.curve_fit(gaus, b2, a1, guess, sigma=(np.zeros(len(b2))+0.01), maxfev=100000)
        assert_allclose(fitParams[2], 1000., atol=0.15)
