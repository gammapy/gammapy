# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from ...stats import (background,
                      background_error,
                      excess,
                      excess_error,
                      significance_on_off,
                      significance,
                      )


def test_docstring_examples():
    """Test the examples given in the docstrings"""
    assert_allclose(background(n_off=4, alpha=0.1), 0.4)
    assert_allclose(background(n_off=9, alpha=0.2), 1.8)

    assert_allclose(background_error(n_off=4, alpha=0.1), 0.2)
    assert_allclose(background_error(n_off=9, alpha=0.2), 0.6)

    assert_allclose(excess(n_on=10, n_off=20, alpha=0.1), 8.0)
    assert_allclose(excess(n_on=4, n_off=9, alpha=0.5), -0.5)

    assert_allclose(excess_error(n_on=10, n_off=20, alpha=0.1), 3.1937439)
    assert_allclose(excess_error(n_on=4, n_off=9, alpha=0.5), 2.5)

    result = significance_on_off(n_on=10, n_off=20, alpha=0.1, method='lima')
    assert_allclose(result, 3.6850322025333071)
    result = significance_on_off(n_on=4, n_off=9, alpha=0.5, method='lima')
    assert_allclose(result, -0.19744427645023557)

    result = significance_on_off(n_on=10, n_off=20, alpha=0.1, method='simple')
    assert_allclose(result, 2.5048971643405982)
    result = significance_on_off(n_on=4, n_off=9, alpha=0.5, method='simple')
    assert_allclose(result, -0.2)

    result = significance_on_off(n_on=10, n_off=20, alpha=0.1, method='lima')
    assert_allclose(result, 3.6850322025333071)

    # Check that the Li & Ma limit formula is correct
    actual = significance(n_observed=1300, mu_background=1100, method='lima')
    assert_allclose(actual, 5.8600870406703329)
    actual = significance_on_off(n_on=1300, n_off=1100 / 1.e-8, alpha=1e-8, method='lima')
    assert_allclose(actual, 5.8600864348078519)


@pytest.mark.xfail
def test_sensitivity():
    """Test if the sensitivity function is the inverse of the significance function."""
    n_ons = np.arange(0.1, 10, 0.3)
    n_offs = np.arange(0.1, 10, 0.3)
    alphas = np.array([1e-3, 1e-2, 0.1, 1, 10])
    for n_on in n_ons:
        for n_off in n_offs:
            for alpha in alphas:
                for method in ['simple', 'lima']:
                    significance = significance_on_off(n_on, n_off, alpha, method=method)
                    excess = sensitivity_on_off(n_off, alpha, significance, method=method)
                    n_on2 = excess + alpha * n_off
                    #print(n_on, n_off, alpha, significance, excess, n_on2)
                    assert_allclose(n_on, n_on2, decimal=3)
