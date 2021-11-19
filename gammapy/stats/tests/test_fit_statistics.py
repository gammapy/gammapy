# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from gammapy import stats


@pytest.fixture
def test_data():
    """Test data for fit statistics tests"""
    test_data = dict(
        mu_sig=[
            0.59752422,
            9.13666449,
            12.98288095,
            5.56974565,
            13.52509804,
            11.81725635,
            0.47963765,
            11.17708176,
            5.18504894,
            8.30202394,
        ],
        n_on=[0, 13, 7, 5, 11, 16, 0, 9, 3, 12],
        n_off=[0, 7, 4, 0, 18, 7, 1, 5, 12, 25],
        alpha=[
            0.83746243,
            0.17003354,
            0.26034507,
            0.69197751,
            0.89557033,
            0.34068848,
            0.0646732,
            0.86411967,
            0.29087245,
            0.74108241,
        ],
    )

    test_data["staterror"] = np.sqrt(test_data["n_on"])

    return test_data


@pytest.fixture
def reference_values():
    """Reference values for fit statistics test.

    Produced using sherpa stats module in dev/sherpa/stats/compare_wstat.py
    """
    return dict(
        wstat=[
            1.19504844,
            0.625311794002,
            4.25810886127,
            0.0603765381044,
            11.7285002468,
            0.206014834301,
            1.084611,
            2.72972381792,
            4.60602990838,
            7.51658734973,
        ],
        cash=[
            1.19504844,
            -39.24635098872072,
            -9.925081055136996,
            -6.034002586236575,
            -30.249839537105466,
            -55.39143500383233,
            0.9592753,
            -21.095413867175516,
            0.49542219758430406,
            -34.19193611846045,
        ],
        cstat=[
            1.19504844,
            1.4423323052792387,
            3.3176610316373925,
            0.06037653810442922,
            0.5038564644586838,
            1.3314041078406706,
            0.9592753,
            0.4546285248764317,
            1.0870959295929628,
            1.4458234764515652,
        ],
    )


def test_wstat(test_data, reference_values):
    statsvec = stats.wstat(
        n_on=test_data["n_on"],
        mu_sig=test_data["mu_sig"],
        n_off=test_data["n_off"],
        alpha=test_data["alpha"],
        extra_terms=True,
    )

    assert_allclose(statsvec, reference_values["wstat"])


def test_cash(test_data, reference_values):
    statsvec = stats.cash(n_on=test_data["n_on"], mu_on=test_data["mu_sig"])
    assert_allclose(statsvec, reference_values["cash"])


def test_cstat(test_data, reference_values):
    statsvec = stats.cstat(n_on=test_data["n_on"], mu_on=test_data["mu_sig"])
    assert_allclose(statsvec, reference_values["cstat"])


def test_cash_sum_cython(test_data):
    counts = np.array(test_data["n_on"], dtype=float)
    npred = np.array(test_data["mu_sig"], dtype=float)
    stat = stats.cash_sum_cython(counts=counts, npred=npred)
    ref = stats.cash(counts, npred).sum()
    assert_allclose(stat, ref)


def test_cash_bad_truncation():
    with pytest.raises(ValueError):
        stats.cash(10, 10, 0.0)


def test_cstat_bad_truncation():
    with pytest.raises(ValueError):
        stats.cstat(10, 10, 0.0)


def test_wstat_corner_cases():
    """test WSTAT formulae for corner cases"""
    n_on = 0
    n_off = 5
    mu_sig = 2.3
    alpha = 0.5

    actual = stats.wstat(n_on=n_on, mu_sig=mu_sig, n_off=n_off, alpha=alpha)
    desired = 2 * (mu_sig + n_off * np.log(1 + alpha))
    assert_allclose(actual, desired)

    actual = stats.get_wstat_mu_bkg(n_on=n_on, mu_sig=mu_sig, n_off=n_off, alpha=alpha)
    desired = n_off / (alpha + 1)
    assert_allclose(actual, desired)

    # n_off = 0 and mu_sig < n_on * (alpha / alpha + 1)
    n_on = 9
    n_off = 0
    mu_sig = 2.3
    alpha = 0.5

    actual = stats.wstat(n_on=n_on, mu_sig=mu_sig, n_off=n_off, alpha=alpha)
    desired = -2 * (mu_sig * (1.0 / alpha) + n_on * np.log(alpha / (1 + alpha)))
    assert_allclose(actual, desired)

    actual = stats.get_wstat_mu_bkg(n_on=n_on, mu_sig=mu_sig, n_off=n_off, alpha=alpha)
    desired = n_on / (1 + alpha) - (mu_sig / alpha)
    assert_allclose(actual, desired)

    # n_off = 0 and mu_sig > n_on * (alpha / alpha + 1)
    n_on = 5
    n_off = 0
    mu_sig = 5.3
    alpha = 0.5

    actual = stats.wstat(n_on=n_on, mu_sig=mu_sig, n_off=n_off, alpha=alpha)
    desired = 2 * (mu_sig + n_on * (np.log(n_on) - np.log(mu_sig) - 1))
    assert_allclose(actual, desired)

    actual = stats.get_wstat_mu_bkg(n_on=n_on, mu_sig=mu_sig, n_off=n_off, alpha=alpha)
    assert_allclose(actual, 0)
