import unittest
import numpy as np
from numpy.testing import assert_almost_equal
from .. import on_off

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
