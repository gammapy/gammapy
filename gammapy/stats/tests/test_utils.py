# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
from ...stats import cov_to_corr


def test_cov_to_corr():
    """Test cov_to_corr against numpy"""
    x = np.array([[0, 2], [1, 1], [2, 0]]).T
    covariance = np.cov(x)
    correlation = cov_to_corr(covariance)
    assert_allclose(correlation, np.corrcoef(x), rtol=1e-12)
