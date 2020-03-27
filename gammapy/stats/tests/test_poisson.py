# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from gammapy.stats import (
    excess_matching_significance,
    excess_matching_significance_on_off,
    excess_ul_helene,
)


def test_excess_ul_helene():
    # The reference values here are from the HESS software
    # TODO: change to reference values from the Helene paper
    excess = excess_ul_helene(excess=50, excess_error=40, significance=3)
    assert_allclose(excess, 171.353908, rtol=1e-3)

    excess = excess_ul_helene(excess=10, excess_error=6, significance=2)
    assert_allclose(excess, 22.123334, rtol=1e-3)

    excess = excess_ul_helene(excess=-23, excess_error=8, significance=3)
    assert_allclose(excess, 13.372179, rtol=1e-3)

    # Check in the very high, Gaussian signal limit, where you have
    # 10000 photons with Poisson noise and no background.
    excess = excess_ul_helene(excess=10000, excess_error=100, significance=1)
    assert_allclose(excess, 10100, atol=0.1)



# TODO: tests should be improved to also cover edge cases,
# similarly to the tests we have for excess_matching_significance_on_off
def test_excess_matching_significance():
    actual = excess_matching_significance(mu_bkg=100, significance=5, method="simple")
    assert_allclose(actual, 50)

    actual = excess_matching_significance(mu_bkg=100, significance=5, method="lima")
    assert_allclose(actual, 54.012755, atol=1e-3)

    # Negative significance should work
    excess = excess_matching_significance(mu_bkg=100, significance=-5, method="simple")
    assert_allclose(excess, -50, atol=1e-3)
    excess = excess_matching_significance(mu_bkg=100, significance=-5, method="lima")
    assert_allclose(excess, -45.631273, atol=1e-3)

    # Cases that can't be achieved with n_on >= 0 should return NaN
    excess = excess_matching_significance(mu_bkg=1, significance=-2)
    assert np.isnan(excess)

    # Arrays should work
    excess = excess_matching_significance(mu_bkg=[1, 2], significance=5)
    assert_allclose(excess, [8.327276, 10.550546], atol=1e-3)


def test_excess_matching_significance_on_off():
    # Negative significance should work
    excess = excess_matching_significance_on_off(n_off=10, alpha=0.1, significance=-1)
    assert_allclose(excess, -0.83198, atol=1e-3)

    # Cases that can't be achieved with n_on >= 0 should return NaN
    excess = excess_matching_significance_on_off(n_off=10, alpha=0.1, significance=-2)
    assert np.isnan(excess)

    # Arrays should work
    excess = excess_matching_significance_on_off(
        n_off=[10, 20], alpha=0.1, significance=5
    )
    assert_allclose(excess, [9.82966, 12.038423], atol=1e-3)
    excess = excess_matching_significance_on_off(
        n_off=[10, 20], alpha=0.1, significance=5, method="simple"
    )
    assert_allclose(excess, [26.05544, 27.03444], atol=1e-3)
    excess = excess_matching_significance_on_off(
        n_off=10, alpha=[0.1, 0.3], significance=5
    )
    assert_allclose(excess, [9.82966, 16.664516], atol=1e-3)
    excess = excess_matching_significance_on_off(
        n_off=10, alpha=0.1, significance=[3, 5]
    )
    assert_allclose(excess, [4.818497, 9.82966], atol=1e-3)
    excess = excess_matching_significance_on_off(
        n_off=[10, 20], alpha=[0.1, 0.3], significance=[3, 5]
    )
    assert_allclose(excess, [4.818497, 20.68810], atol=1e-3)
    excess = excess_matching_significance_on_off(
        n_off=[[10, 20], [10, 20]], alpha=0.1, significance=5
    )
    assert_allclose(excess, [[9.82966, 12.038423], [9.82966, 12.038423]], atol=1e-3)

