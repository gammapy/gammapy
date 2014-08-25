# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from ...stats import s_to_p, cov_to_corr

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@pytest.mark.skipif('not HAS_SCIPY')
def test_s_to_p():
    p = s_to_p(5)
    assert_allclose(p, 2.8665157187919328e-07)


def test_cov_to_corr():
    """Test cov_to_corr against numpy"""
    x = np.array([[0, 2], [1, 1], [2, 0]]).T
    covariance = np.cov(x)
    correlation = cov_to_corr(covariance)
    assert_allclose(correlation, np.corrcoef(x), rtol=1E-12)
