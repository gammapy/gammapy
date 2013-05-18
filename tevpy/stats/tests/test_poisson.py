import numpy as np
from numpy.testing import assert_almost_equal
from .. import poisson

def test_docstring_examples():
    """Test the examples given in the docstrings"""
    assert_almost_equal(poisson.background(n_off=4, alpha=0.1), 0.4)
    assert_almost_equal(poisson.background(n_off=9, alpha=0.2), 1.8)

    assert_almost_equal(poisson.background_error(n_off=4, alpha=0.1), 0.2)
    assert_almost_equal(poisson.background_error(n_off=9, alpha=0.2), 0.6)

    assert_almost_equal(poisson.excess(n_on=10, n_off=20, alpha=0.1), 8.0)
    assert_almost_equal(poisson.excess(n_on=4, n_off=9, alpha=0.5), -0.5)

    assert_almost_equal(poisson.excess_error(n_on=10, n_off=20, alpha=0.1), 3.1937439)
    assert_almost_equal(poisson.excess_error(n_on=4, n_off=9, alpha=0.5), 2.5)

    result = poisson.significance_on_off(n_on=10, n_off=20, alpha=0.1, method='lima')
    assert_almost_equal(result, 3.6850322025333071)
    result = poisson.significance_on_off(n_on=4, n_off=9, alpha=0.5, method='lima')
    assert_almost_equal(result, -0.19744427645023557)

    result = poisson.significance_on_off(n_on=10, n_off=20, alpha=0.1, method='simple')
    assert_almost_equal(result, 2.5048971643405982)
    result = poisson.significance_on_off(n_on=4, n_off=9, alpha=0.5, method='simple')
    assert_almost_equal(result, -0.2)

    result = poisson.significance_on_off(n_on=10, n_off=20, alpha=0.1, method='lima')
    assert_almost_equal(result, 3.6850322025333071)

    # Check that the Li & Ma limit formula is correct
    assert_almost_equal(poisson.significance(n_observed=1300, mu_background=1100, method='lima'), 5.8600870406703329)
    assert_almost_equal(poisson.significance_on_off(n_on=1300, n_off=1100 / 1.e-8, alpha=1e-8, method='lima'), 5.8600864348078519)

    
'''
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
                		in zip([poisson.significance, poisson.significance_simple],
						[poisson.sensitivity_lima, poisson.sensitivity_simple]):
                        s = significance_func(on, off, alpha)
                        excess = sensitivity_func(off, alpha, s)
                        on_reco = excess + alpha * off
                        #print on, off, alpha, s, excess, on_reco
                        # TODO
                        #assert_almost_equal(on, on_reco, decimal=3)
'''