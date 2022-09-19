# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from gammapy.stats import CashCountsStatistic, WStatCountsStatistic

ref_array = np.ones((3, 2, 4))

values = [
    (1, 2, [-1.0, -0.78339367, 0.216698]),
    (5, 1, [4.0, 2.84506224, 2.220137e-3]),
    (10, 5, [5.0, 1.96543726, 0.024682]),
    (100, 23, [77.0, 11.8294207, 1.37e-32]),
    (1, 20, [-19, -5.65760863, 7.5e-09]),
    (5 * ref_array, 1 * ref_array, [4.0, 2.84506224, 2.220137e-3]),
]


@pytest.mark.parametrize(("n_on", "mu_bkg", "result"), values)
def test_cash_basic(n_on, mu_bkg, result):
    stat = CashCountsStatistic(n_on, mu_bkg)
    excess = stat.n_sig
    sqrt_ts = stat.sqrt_ts
    p_value = stat.p_value

    assert_allclose(excess, result[0])
    assert_allclose(sqrt_ts, result[1], atol=1e-5)
    assert_allclose(p_value, result[2], atol=1e-5)


values = [
    (1, 2, [0.69829, 1.35767667]),
    (5, 1, [1.915916, 2.581106]),
    (10, 5, [2.838105, 3.504033]),
    (100, 23, [9.669482, 10.336074]),
    (1, 20, [0.69829, 1.357677]),
    (5 * ref_array, 1 * ref_array, [1.915916, 2.581106]),
]


@pytest.mark.parametrize(("n_on", "mu_bkg", "result"), values)
def test_cash_errors(n_on, mu_bkg, result):
    stat = CashCountsStatistic(n_on, mu_bkg)
    errn = stat.compute_errn()
    errp = stat.compute_errp()

    assert_allclose(errn, result[0], atol=1e-5)
    assert_allclose(errp, result[1], atol=1e-5)


values = [
    (1, 2, [5.517193]),
    (5, 1, [13.98959]),
    (10, 5, [17.696064]),
    (100, 23, [110.07206]),
    (1, 20, [-12.482807]),
    (5 * ref_array, 1 * ref_array, [13.98959]),
]


@pytest.mark.parametrize(("n_on", "mu_bkg", "result"), values)
def test_cash_ul(n_on, mu_bkg, result):
    stat = CashCountsStatistic(n_on, mu_bkg)
    ul = stat.compute_upper_limit()

    assert_allclose(ul, result[0], atol=1e-5)


values = [
    (100, 5, 54.012755),
    (100, -5, -45.631273),
    pytest.param(1, -2, np.nan, marks=pytest.mark.xfail),
    ([1, 2], 5, [8.327276, 10.550546]),
]


@pytest.mark.parametrize(("mu_bkg", "significance", "result"), values)
def test_cash_excess_matching_significance(mu_bkg, significance, result):
    stat = CashCountsStatistic(0, mu_bkg)
    excess = stat.n_sig_matching_significance(significance)

    assert_allclose(excess, result, atol=1e-3)


values = [
    (1, 2, 1, [-1.0, -0.5829220133009171, 0.279973]),
    (5, 1, 1, [4.0, 1.7061745691234782, 0.043988]),
    (10, 5, 0.3, [8.5, 3.5853812867949024, 1.68293e-4]),
    (10, 23, 0.1, [7.7, 3.443415522820395, 2.87208e-4]),
    (1, 20, 1.0, [-19, -4.590373638528086, 2.2122e-06]),
    (
        5 * ref_array,
        1 * ref_array,
        1 * ref_array,
        [4.0, 1.7061745691234782, 0.043988],
    ),
]


@pytest.mark.parametrize(("n_on", "n_off", "alpha", "result"), values)
def test_wstat_basic(n_on, n_off, alpha, result):
    stat = WStatCountsStatistic(n_on, n_off, alpha)
    excess = stat.n_sig
    sqrt_ts = stat.sqrt_ts
    p_value = stat.p_value

    assert_allclose(excess, result[0], rtol=1e-4)
    assert_allclose(sqrt_ts, result[1], rtol=1e-4)
    assert_allclose(p_value, result[2], rtol=1e-4)


values = [
    (5, 1, 1, 3, [1, 0.422261, 0.336417, 0.178305]),
    (5, 1, 1, 1, [3.0, 1.29828, 0.097095, 1.685535]),
    (5, 1, 1, 6, [-2, -0.75585, 0.224869, 0.571311]),
]


@pytest.mark.parametrize(("n_on", "n_off", "alpha", "mu_sig", "result"), values)
def test_wstat_with_musig(n_on, n_off, alpha, mu_sig, result):

    stat = WStatCountsStatistic(n_on, n_off, alpha, mu_sig)
    excess = stat.n_sig
    sqrt_ts = stat.sqrt_ts
    p_value = stat.p_value
    del_ts = stat.ts

    assert_allclose(excess, result[0], rtol=1e-4)
    assert_allclose(sqrt_ts, result[1], rtol=1e-4)
    assert_allclose(p_value, result[2], rtol=1e-4)
    assert_allclose(del_ts, result[3], rtol=1e-4)


values = [
    (1, 2, 1, [1.942465, 1.762589]),
    (5, 1, 1, [2.310459, 2.718807]),
    (10, 5, 0.3, [2.932472, 3.55926]),
    (10, 23, 0.1, [2.884366, 3.533279]),
    (1, 20, 1.0, [4.897018, 4.299083]),
    (5 * ref_array, 1 * ref_array, 1 * ref_array, [2.310459, 2.718807]),
]


@pytest.mark.parametrize(("n_on", "n_off", "alpha", "result"), values)
def test_wstat_errors(n_on, n_off, alpha, result):
    stat = WStatCountsStatistic(n_on, n_off, alpha)
    errn = stat.compute_errn()
    errp = stat.compute_errp()

    assert_allclose(errn, result[0], atol=1e-5)
    assert_allclose(errp, result[1], atol=1e-5)


values = [
    (1, 2, 1, [6.075534]),
    (5, 1, 1, [14.222831]),
    (10, 5, 0.3, [21.309229]),
    (10, 23, 0.1, [20.45803]),
    (1, 20, 1.0, [-7.078228]),
    (5 * ref_array, 1 * ref_array, 1 * ref_array, [14.222831]),
]


@pytest.mark.parametrize(("n_on", "n_off", "alpha", "result"), values)
def test_wstat_ul(n_on, n_off, alpha, result):
    stat = WStatCountsStatistic(n_on, n_off, alpha)
    ul = stat.compute_upper_limit()

    assert_allclose(ul, result[0], rtol=1e-5)


values = [
    ([10, 20], [0.1, 0.1], 5, [9.82966, 12.0384229]),
    ([10, 10], [0.1, 0.3], 5, [9.82966, 16.664516]),
    ([10], [0.1], 3, [4.818497]),
    (
        [[10, 20], [10, 20]],
        [[0.1, 0.1], [0.1, 0.1]],
        5,
        [[9.82966, 12.129523], [9.82966, 12.129523]],
    ),
]


@pytest.mark.parametrize(("n_off", "alpha", "significance", "result"), values)
def test_wstat_excess_matching_significance(n_off, alpha, significance, result):
    stat = WStatCountsStatistic(0, n_off, alpha)
    excess = stat.n_sig_matching_significance(significance)

    assert_allclose(excess, result, rtol=1e-2)


def test_cash_sum():
    on = [1, 2, 3]
    bkg = [0.5, 0.7, 1.3]

    stat = CashCountsStatistic(on, bkg)
    stat_sum = stat.sum()

    assert stat_sum.n_on == 6
    assert stat_sum.n_bkg == 2.5

    new_size = (2, 10, 3)
    on = np.resize(on, new_size)
    bkg = np.resize(bkg, new_size)

    stat = CashCountsStatistic(on, bkg)
    stat_sum = stat.sum(axis=(2))

    assert stat_sum.n_on.shape == (2, 10)
    assert_allclose(stat_sum.n_on, 6)
    assert_allclose(stat_sum.n_bkg, 2.5)

    stat_sum = stat.sum(axis=(0, 1))

    assert stat_sum.n_on.shape == (3,)
    assert_allclose(stat_sum.n_on, (20, 40, 60))
    assert_allclose(stat_sum.n_bkg, (10, 14, 26))


def test_wstat_sum():
    on = [1, 2, 3]
    off = [5, 14, 8]
    alpha = [0.1, 0.05, 0.1625]

    stat = WStatCountsStatistic(on, off, alpha)
    stat_sum = stat.sum()

    assert stat_sum.n_on == 6
    assert stat_sum.n_off == 27
    assert stat_sum.n_bkg == 2.5
    assert_allclose(stat_sum.alpha, 0.0925925925925925)

    new_size = (2, 10, 3)
    off = np.resize(off, new_size)
    on = np.resize(on, new_size)
    alpha = np.resize(alpha, new_size)

    stat = WStatCountsStatistic(on, off, alpha)
    stat_sum = stat.sum(axis=(2))

    assert stat_sum.n_on.shape == (2, 10)
    assert_allclose(stat_sum.n_on, 6)
    assert_allclose(stat_sum.n_bkg, 2.5)
    assert_allclose(stat_sum.alpha, 0.0925925925925925)

    stat_sum = stat.sum(axis=(0, 1))

    assert stat_sum.n_on.shape == (3,)
    assert_allclose(stat_sum.n_on, (20, 40, 60))
    assert_allclose(stat_sum.n_bkg, (10, 14, 26))
