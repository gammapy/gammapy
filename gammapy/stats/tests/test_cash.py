# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
from gammapy.stats import CashEstimator

values = [
    (1, 2,    [-1., -0.78339367]),
    (5, 1,    [4., 2.84506224]),
    (10, 5,   [5., 1.96543726]),
    (100, 23, [77., 11.8294207]),
    (1, 20,   [-19, -5.65760863])
]

@pytest.mark.parametrize(("n_on", "mu_bkg", "result"), values)
def test_cash_basic(n_on, mu_bkg, result):
    stat = CashEstimator(n_on, mu_bkg)
    excess = stat.excess
    significance = stat.significance

    assert_allclose(excess, result[0])
    assert_allclose(significance, result[1], atol=1e-5)

values = [
    (1, 2,    [-1.,1.35767667]),
    (5, 1,    [-1.915916, 2.581106]),
    (10, 5,   [-2.838105, 3.504033]),
    (100, 23, [-9.669482, 10.336074]),
    (1, 20,   [-1, 1.357677]),
]

@pytest.mark.parametrize(("n_on", "mu_bkg", "result"), values)
def test_cash_errors(n_on, mu_bkg, result):
    stat = CashEstimator(n_on, mu_bkg)
    errn = stat.compute_errn()
    errp = stat.compute_errp()

    assert_allclose(errn, result[0], atol=1e-5)
    assert_allclose(errp, result[1], atol=1e-5)

values = [
    (1, 2,    [2.254931]),
    (5, 1,    [8.931234]),
    (10, 5,   [11.519662]),
    (100, 23, [95.334612]),
    (1, 20,   [1.575842]),
]

@pytest.mark.parametrize(("n_on", "mu_bkg", "result"), values)
def test_cash_ul(n_on, mu_bkg, result):
    stat = CashEstimator(n_on, mu_bkg)
    ul = stat.compute_upper_limit()

    assert_allclose(ul, result[0], atol=1e-5)
