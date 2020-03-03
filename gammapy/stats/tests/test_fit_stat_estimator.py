# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from gammapy.stats import CashCountsStatistic, WStatCountsStatistic

ref_array = np.ones((3, 2, 4))

values = [
    (1, 2, [-1.0, -0.78339367]),
    (5, 1, [4.0, 2.84506224]),
    (10, 5, [5.0, 1.96543726]),
    (100, 23, [77.0, 11.8294207]),
    (1, 20, [-19, -5.65760863]),
    (5 * ref_array, 1 * ref_array, [4.0, 2.84506224]),
]


@pytest.mark.parametrize(("n_on", "mu_bkg", "result"), values)
def test_cash_basic(n_on, mu_bkg, result):
    stat = CashCountsStatistic(n_on, mu_bkg)
    excess = stat.excess
    significance = stat.significance

    assert_allclose(excess, result[0])
    assert_allclose(significance, result[1], atol=1e-5)


values = [
    (1, 2, [-1.0, 1.35767667]),
    (5, 1, [-1.915916, 2.581106]),
    (10, 5, [-2.838105, 3.504033]),
    (100, 23, [-9.669482, 10.336074]),
    (1, 20, [-1, 1.357677]),
    (5 * ref_array, 1 * ref_array, [-1.915916, 2.581106]),
]


@pytest.mark.parametrize(("n_on", "mu_bkg", "result"), values)
def test_cash_errors(n_on, mu_bkg, result):
    stat = CashCountsStatistic(n_on, mu_bkg)
    errn = stat.compute_errn()
    errp = stat.compute_errp()

    assert_allclose(errn, result[0], atol=1e-5)
    assert_allclose(errp, result[1], atol=1e-5)


values = [
    (1, 2, [2.254931]),
    (5, 1, [8.931234]),
    (10, 5, [11.519662]),
    (100, 23, [95.334612]),
    (1, 20, [1.575842]),
    (5 * ref_array, 1 * ref_array, [8.931234]),
]


@pytest.mark.parametrize(("n_on", "mu_bkg", "result"), values)
def test_cash_ul(n_on, mu_bkg, result):
    stat = CashCountsStatistic(n_on, mu_bkg)
    ul = stat.compute_upper_limit()

    assert_allclose(ul, result[0], atol=1e-5)


values = [
    (1, 2, 1, [-1.0, -0.5829220133009171]),
    (5, 1, 1, [4.0, 1.7061745691234782]),
    (10, 5, 0.3, [8.5, 3.5853812867949024]),
    (10, 23, 0.1, [7.7, 3.443415522820395]),
    (1, 20, 1.0, [-19, -4.590373638528086]),
    (5 * ref_array, 1 * ref_array, 1 * ref_array, [4.0, 1.7061745691234782]),
]


@pytest.mark.parametrize(("n_on", "n_off", "alpha", "result"), values)
def test_wstat_basic(n_on, n_off, alpha, result):
    stat = WStatCountsStatistic(n_on, n_off, alpha)
    excess = stat.excess
    significance = stat.significance

    assert_allclose(excess, result[0])
    assert_allclose(significance, result[1], atol=1e-5)


values = [
    (1, 2, 1, [-1.942465, 1.762589]),
    (5, 1, 1, [-2.310459, 2.718807]),
    (10, 5, 0.3, [-2.932472, 3.55926]),
    (10, 23, 0.1, [-2.884366, 3.533279]),
    (1, 20, 1.0, [-4.897018, 4.299083]),
    (5 * ref_array, 1 * ref_array, 1 * ref_array, [-2.310459, 2.718807]),
]


@pytest.mark.parametrize(("n_on", "n_off", "alpha", "result"), values)
def test_wstat_errors(n_on, n_off, alpha, result):
    stat = WStatCountsStatistic(n_on, n_off, alpha)
    errn = stat.compute_errn()
    errp = stat.compute_errp()

    assert_allclose(errn, result[0], atol=1e-5)
    assert_allclose(errp, result[1], atol=1e-5)


values = [
    (1, 2, 1, [2.57939]),
    (5, 1, 1, [9.118247]),
    (10, 5, 0.3, [15.10175]),
    (10, 23, 0.1, [14.263802]),
    (1, 20, 1.0, [1.646071]),
    (5 * ref_array, 1 * ref_array, 1 * ref_array, [9.118247]),
]


@pytest.mark.parametrize(("n_on", "n_off", "alpha", "result"), values)
def test_wstat_ul(n_on, n_off, alpha, result):
    stat = WStatCountsStatistic(n_on, n_off, alpha)
    ul = stat.compute_upper_limit()

    assert_allclose(ul, result[0], atol=1e-5)
