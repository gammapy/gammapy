import unittest
import numpy as np
from numpy.testing import assert_almost_equal
from .. import on_off

def test_docstring_examples():
    """Test the examples given in the docstrings"""
    assert_almost_equal(on_off.background(n_off=4, alpha=0.1), 0.4)
    assert_almost_equal(on_off.background(n_off=9, alpha=0.2), 1.8)

    assert_almost_equal(on_off.background_error(n_off=4, alpha=0.1), 0.2)
    assert_almost_equal(on_off.background_error(n_off=9, alpha=0.2), 0.6)

    assert_almost_equal(on_off.excess(n_on=10, n_off=20, alpha=0.1), 8.0)
    assert_almost_equal(on_off.excess(n_on=4, n_off=9, alpha=0.5), -0.5)

    assert_almost_equal(on_off.excess_error(n_on=10, n_off=20, alpha=0.1), 3.1937439)
    assert_almost_equal(on_off.excess_error(n_on=4, n_off=9, alpha=0.5), 2.5)

    assert_almost_equal(on_off.significance_lima(n_on=10, n_off=20, alpha=0.1), 3.6850322025333071)
    assert_almost_equal(on_off.significance_lima(n_on=4, n_off=9, alpha=0.5), -0.19744427645023557)

    assert_almost_equal(on_off.significance_lima_limit(n_on=1300, mu_background=1100), 5.8600870406703329)
    assert_almost_equal(on_off.significance_lima(n_on=1300, n_off=1100 / 1.e-8, alpha=1e-8), 5.8600864348078519)

    assert_almost_equal(on_off.significance_simple(n_on=10, n_off=20, alpha=0.1), 2.5048971643405982)
    assert_almost_equal(on_off.significance_simple(n_on=4, n_off=9, alpha=0.5), -0.2)

    assert_almost_equal(on_off.significance(n_on=10, n_off=20, alpha=0.1, method='lima'), 3.6850322025333071)
    

class TestSignificance(unittest.TestCase):
    def test_LiMa(self):
        """ Test the HESS against the fast LiMa function """
        # TODO: generate a CSV reference results file
        #ons = np.arange(0.1, 10, 0.3)
        #offs = np.arange(0.1, 10, 0.3)
        #alphas = np.array([1e-3, 1e-2, 0.1, 1, 10])
    def _test_sensitivity(self):
        """ Test if the sensitivity function is the
        inverse of the significance function """
        ons = np.arange(0.1, 10, 0.3)
        offs = np.arange(0.1, 10, 0.3)
        alphas = np.array([1e-3, 1e-2, 0.1, 1, 10])
        for on in ons:
            for off in offs:
                for alpha in alphas:
                    for significance_func, sensitivity_func \
                		in zip([on_off.significance, on_off.significance_simple],
						[on_off.sensitivity_lima, on_off.sensitivity_simple]):
                        s = significance_func(on, off, alpha)
                        excess = sensitivity_func(off, alpha, s)
                        on_reco = excess + alpha * off
                        #print on, off, alpha, s, excess, on_reco
                        # TODO
                        #assert_almost_equal(on, on_reco, decimal=3)
