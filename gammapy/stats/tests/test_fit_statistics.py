# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from ...utils.testing import requires_dependency
from ...utils.random import get_random_state
from ... import stats as gammapy_stats


# TODO; change to a test dataset that doesn't use random values (just list the numbers)
# and use `pytest.fixture` to use it in the tests.
def get_test_data():
    random_state = get_random_state(3)
    # put factor to 100 to not run into special cases in WStat
    model = random_state.rand(10) * 100
    data = random_state.poisson(model)
    staterror = np.sqrt(data)
    off_vec = random_state.poisson(0.7 * model)
    return data, model, staterror, off_vec


# TODO : Produce reference numbers outside of test (avoid sherpa dependency)
# Note: There is an independent implementation of the XSPEC  wstat that can
# be used for debugging: gammapy/dev/sherpa/stats/xspec_stats.py    
# Also there is the script dev/sherpa/stats/compare_stats.py that is very
# usefull for debugging

@requires_dependency('sherpa')
def test_cstat():
    import sherpa.stats as ss
    sherpa_stat = ss.CStat()
    data, model, staterror, off_vec = get_test_data()
    desired, fvec = sherpa_stat.calc_stat(data, model, staterror=staterror)

    statsvec = gammapy_stats.cstat(n_on=data, mu_on=model)
    actual = np.sum(statsvec)
    assert_allclose(actual, desired)


@requires_dependency('sherpa')
def test_cash():
    import sherpa.stats as ss
    sherpa_stat = ss.Cash()
    data, model, staterror, off_vec = get_test_data()
    desired, fvec = sherpa_stat.calc_stat(data, model, staterror=staterror)

    statsvec = gammapy_stats.cash(n_on=data, mu_on=model)
    actual = np.sum(statsvec)
    assert_allclose(actual, desired)


# TODO: Make this test independent of Sherpa
@requires_dependency('sherpa')
def test_wstat():
    import sherpa.stats as ss
    sherpa_stat = ss.WStat()
    data, model, staterror, off_vec = get_test_data()
    alpha = np.ones(len(data)) * 0.2

    statsvec = gammapy_stats.wstat(n_on=data,
                                   mu_sig=model,
                                   n_off=off_vec,
                                   alpha=alpha,
                                   extra_terms=True)

    # This is how sherpa wants the background (found by trial and error)
    bkg = dict(bkg=off_vec,
               exposure_time=[1, 1],
               backscale_ratio=1. / alpha,
               data_size=len(data)
               )

    # Check for one bin first
    test_bin = 0
    bkg_testbin = dict(bkg=off_vec[test_bin],
                       exposure_time=[1, 1],
                       backscale_ratio=1. / alpha[test_bin],
                       data_size=1)

    desired_testbin, fvec = sherpa_stat.calc_stat(data[test_bin],
                                                  model[test_bin],
                                                  staterror=staterror[test_bin],
                                                  bkg=bkg_testbin)

    actual_testbin = statsvec[test_bin]
    #    assert_allclose(actual_testbin, desired_testbin)

    # Now check total stat for all bins
    desired, fvec = sherpa_stat.calc_stat(data, model, staterror=staterror,
                                          bkg=bkg)

    actual = np.sum(statsvec)
    print(fvec)
    assert_allclose(actual, desired)
