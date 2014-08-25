# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from ...stats import (cash,
                      cstat,
                      chi2datavar,
                      )

# TODO: test against Sherpa results
# see scripts here: https://github.com/gammapy/gammapy/tree/master/dev/sherpa/stats

@pytest.mark.xfail
def test_likelihood_stats():
    # Create some example input
    M = np.array([[0.2, 0.5, 1, 2], [10, 10.5, 100.5, 1000]])
    np.random.seed(0)
    D = np.random.poisson(M)

    assert_allclose(cash(D, M), 42)
    assert_allclose(cstat(D, M), 42)


@pytest.mark.xfail
def test_chi2_stats():
    A_S = np.array([[0, 0.5, 1, 2], [10, 10.5, 100.5, 1000]])
    A_B = A_S
    N_S = A_S
    N_B = A_S
    actual = chi2datavar(N_S, N_B, A_S, A_B)
    assert_allclose(actual, 43)
