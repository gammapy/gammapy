# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
import pytest
from ...utils.testing import requires_dependency
from ... import stats


@pytest.fixture
def test_data():
    """Test data for fit statistics tests"""
    test_data = dict(
        mu_sig=np.array([0.59752422, 9.13666449, 12.98288095, 5.56974565,
                         13.52509804, 11.81725635, 0.47963765, 11.17708176,
                         5.18504894, 8.30202394]),
        n_on=np.array([0, 13, 7, 5, 11, 16, 0, 9, 3, 12]),
        n_off=np.array([0, 7, 4, 0, 18, 7, 1, 5, 12, 25]),
        alpha=np.array([0.83746243, 0.17003354, 0.26034507, 0.69197751,
                        0.89557033, 0.34068848, 0.0646732, 0.86411967,
                        0.29087245, 0.74108241])
    )

    test_data['staterror'] = np.sqrt(test_data['n_on'])

    return test_data


@pytest.fixture
def reference_values():
    """Reference values for fit statistics test.

    Produced using sherpa stats module in dev/sherpa/stats/compare_wstat.py
    """
    return dict(
        wstat=[1.19504844, 0.625311794002, 4.25810886127, 0.0603765381044,
               11.7285002468, 0.206014834301, 1.084611, 2.72972381792,
               4.60602990838, 7.51658734973]
    )


@pytest.mark.xfail(reason='sherpa implementation changed')
@requires_dependency('sherpa')
def test_cstat(test_data):
    import sherpa.stats as ss
    sherpa_stat = ss.CStat()
    data = test_data['n_on']
    model = test_data['mu_sig']
    staterror = test_data['staterror']
    desired, fvec = sherpa_stat.calc_stat(data, model, staterror=staterror)

    statsvec = stats.cstat(n_on=data, mu_on=model)
    actual = np.sum(statsvec)
    assert_allclose(actual, desired)


@pytest.mark.xfail(reason='sherpa implementation changed')
@requires_dependency('sherpa')
def test_cash(test_data):
    import sherpa.stats as ss
    sherpa_stat = ss.Cash()
    data = test_data['n_on']
    model = test_data['mu_sig']
    staterror = test_data['staterror']
    desired, fvec = sherpa_stat.calc_stat(data, model, staterror=staterror)

    statsvec = stats.cash(n_on=data, mu_on=model)
    actual = np.sum(statsvec)
    assert_allclose(actual, desired)


def test_wstat(test_data, reference_values):
    statsvec = stats.wstat(
        n_on=test_data['n_on'],
        mu_sig=test_data['mu_sig'],
        n_off=test_data['n_off'],
        alpha=test_data['alpha'],
        extra_terms=True,
    )

    assert_allclose(statsvec, reference_values['wstat'])


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
    desired = -2 * (mu_sig * (1. / alpha) + n_on * np.log(alpha / (1 + alpha)))
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
