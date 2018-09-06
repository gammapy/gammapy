# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
import pytest
from ...utils.testing import requires_dependency
from ...stats import (
    fc_find_acceptance_interval_gauss,
    fc_find_acceptance_interval_poisson,
    fc_construct_acceptance_intervals_pdfs,
    fc_get_limits,
    fc_fix_limits,
    fc_find_limit,
    fc_find_average_upper_limit,
    fc_construct_acceptance_intervals,
)


@requires_dependency("scipy")
def test_acceptance_interval_gauss():
    sigma = 1
    n_sigma = 10
    n_step = 1000
    cl = 0.90

    x_bins = np.linspace(-n_sigma * sigma, n_sigma * sigma, n_step, endpoint=True)

    # The test reverses a result from the Feldman and Cousins paper. According
    # to Table X, for a measured value of 2.6 the 90% confidence interval should
    # be 1.02 and 4.24. Reversed that means that for mu=1.02, the acceptance
    # interval should end at 2.6 and for mu=4.24 should start at 2.6.
    (x_min, x_max) = fc_find_acceptance_interval_gauss(1.02, sigma, x_bins, cl)
    assert_allclose(x_max, 2.6, atol=0.1)

    (x_min, x_max) = fc_find_acceptance_interval_gauss(4.24, sigma, x_bins, cl)
    assert_allclose(x_min, 2.6, atol=0.1)

    # At mu=0, confidence interval should start at the negative x_bins range.
    (x_min, x_max) = fc_find_acceptance_interval_gauss(0, sigma, x_bins, cl)
    assert_allclose(x_min, -n_sigma * sigma)

    # Pass too few x_bins to reach confidence level.
    x_bins = np.linspace(-sigma, sigma, n_step, endpoint=True)
    with pytest.raises(ValueError):
        fc_find_acceptance_interval_gauss(0, 1, x_bins, cl)


@requires_dependency("scipy")
def test_acceptance_interval_poisson():
    background = 0.5
    n_bins_x = 100
    cl = 0.90

    x_bins = np.arange(0, n_bins_x)

    # The test reverses a result from the Feldman and Cousins paper. According
    # to Table IV, for a measured value of 10 the 90% confidence interval should
    # be 5.00 and 16.00. Reversed that means that for mu=5.0, the acceptance
    # interval should end at 10 and for mu=16.00 should start at 10.
    (x_min, x_max) = fc_find_acceptance_interval_poisson(5.00, background, x_bins, cl)
    assert_allclose(x_max, 10)

    (x_min, x_max) = fc_find_acceptance_interval_poisson(16.00, background, x_bins, cl)
    assert_allclose(x_min, 10)

    # Pass too few x_bins to reach confidence level.
    with pytest.raises(ValueError):
        fc_find_acceptance_interval_poisson(0, 7, x_bins[0:10], cl)


@requires_dependency("scipy")
def test_numerical_confidence_interval_pdfs():
    from scipy import stats

    background = 3.0
    step_width_mu = 0.005
    mu_min = 0
    mu_max = 15
    n_bins_x = 50
    cl = 0.90

    x_bins = np.arange(0, n_bins_x)
    mu_bins = np.linspace(mu_min, mu_max, mu_max / step_width_mu + 1, endpoint=True)

    matrix = [stats.poisson(mu + background).pmf(x_bins) for mu in mu_bins]

    acceptance_intervals = fc_construct_acceptance_intervals_pdfs(matrix, cl)

    lower_limit_num, upper_limit_num, _ = fc_get_limits(
        mu_bins, x_bins, acceptance_intervals
    )

    fc_fix_limits(lower_limit_num, upper_limit_num)

    x_measured = 6
    upper_limit = fc_find_limit(x_measured, upper_limit_num, mu_bins)
    lower_limit = fc_find_limit(x_measured, lower_limit_num, mu_bins)

    # Values are taken from Table IV in the Feldman and Cousins paper.
    assert_allclose(upper_limit, 8.47, atol=0.01)
    assert_allclose(lower_limit, 0.15, atol=0.01)

    # A value which is not inside the x axis range should raise an exception
    with pytest.raises(ValueError):
        fc_find_limit(51, upper_limit_num, mu_bins)

    # Calculate the average upper limit. The upper limit calculated here is
    # only defined for a small x range, so limit the x bins here so the
    # calculation of the average limit is meaningful.
    average_upper_limit = fc_find_average_upper_limit(
        x_bins, matrix, upper_limit_num, mu_bins
    )

    # Values are taken from Table XII in the Feldman and Cousins paper.
    # A higher accuracy would require a higher mu_max, which would increase
    # the computation time.
    assert_allclose(average_upper_limit, 4.42, atol=0.1)


@requires_dependency("scipy")
def test_numerical_confidence_interval_values():
    from scipy import stats

    sigma = 1
    n_sigma = 10
    n_bins_x = 100
    step_width_mu = 0.05
    mu_min = 0
    mu_max = 8
    cl = 0.90

    x_bins = np.linspace(-n_sigma * sigma, n_sigma * sigma, n_bins_x, endpoint=True)
    mu_bins = np.linspace(mu_min, mu_max, mu_max / step_width_mu + 1, endpoint=True)

    distribution_dict = dict(
        (mu, [stats.norm.rvs(loc=mu, scale=sigma, size=5000)]) for mu in mu_bins
    )

    acceptance_intervals = fc_construct_acceptance_intervals(
        distribution_dict, x_bins, cl
    )

    lower_limit_num, upper_limit_num, _ = fc_get_limits(
        mu_bins, x_bins, acceptance_intervals
    )

    fc_fix_limits(lower_limit_num, upper_limit_num)

    x_measured = 1.7
    upper_limit = fc_find_limit(x_measured, upper_limit_num, mu_bins)

    # Value taken from Table X in the Feldman and Cousins paper.
    assert_allclose(upper_limit, 3.34, atol=0.1)
