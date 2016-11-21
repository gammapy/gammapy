# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from ...utils.testing import requires_dependency
from ...utils.random import get_random_state
from ... import stats 


@pytest.fixture
def test_data():
    """Test data for fit statistics tests"""
    test_data = dict(
        mu_sig = np.array([0.59752422, 9.13666449, 12.98288095, 5.56974565,
                           13.52509804, 11.81725635, 0.47963765, 11.17708176,
                           5.18504894, 8.30202394]),
        n_on = np.array([0, 13, 7, 5, 11, 16, 0, 9, 3, 12]),
        n_off = np.array([0, 7, 4, 0, 18, 7, 1, 5, 12, 25]),
        alpha = np.array([0.83746243, 0.17003354, 0.26034507, 0.69197751,
                          0.89557033, 0.34068848, 0.0646732, 0.86411967,
                          0.29087245, 0.74108241])
    )
    
    test_data['staterror'] = np.sqrt(test_data['n_on']),

    return test_data 


@pytest.fixture
def reference_values():
    """Reference values for fit statistics test.

    Produced using sherpa stats module in dev/sherpa/stats/compare_wstat.py
    """
    ref_vals = dict(
        wstat=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    return ref_vals

# TODO : Produce reference numbers outside of test (avoid sherpa dependency)
# Note: There is an independent implementation of the XSPEC  wstat that can
# be used for debugging: gammapy/dev/sherpa/stats/xspec_stats.py
# Also there is the script dev/sherpa/stats/compare_stats.py that is very
# usefull for debugging


#@requires_dependency('sherpa')
#def test_cstat():
#    import sherpa.stats as ss
#    sherpa_stat = ss.CStat()
#    data, model, staterror, off_vec = test_data()
#    desired, fvec = sherpa_stat.calc_stat(data, model, staterror=staterror)
#
#    statsvec = stats.cstat(n_on=data, mu_on=model)
#    actual = np.sum(statsvec)
#    assert_allclose(actual, desired)
#
#
#@requires_dependency('sherpa')
#def test_cash():
#    import sherpa.stats as ss
#    sherpa_stat = ss.Cash()
#    data, model, staterror, off_vec = test_data()
#    desired, fvec = sherpa_stat.calc_stat(data, model, staterror=staterror)
#
#    statsvec = stats.cash(n_on=data, mu_on=model)
#    actual = np.sum(statsvec)
#    assert_allclose(actual, desired)


def test_wstat(test_data, reference_values):
    statsvec = stats.wstat(n_on=test_data['n_on'],
                           mu_sig=test_data['mu_sig'],
                           n_off=test_data['n_off'],
                           alpha=test_data['alpha'],
                           extra_terms=True)

    assert_allclose(statsvec, reference_values['wstat'])

#    # This is how sherpa wants the background (found by trial and error)
#    bkg = dict(bkg=off_vec,
#               exposure_time=[1, 1],
#               backscale_ratio=1. / alpha,
#               data_size=len(data)
#               )
#
#    # Check for one bin first
#    test_bin = 0
#    bkg_testbin = dict(bkg=off_vec[test_bin],
#                       exposure_time=[1, 1],
#                       backscale_ratio=1. / alpha[test_bin],
#                       data_size=1)
#    desired_testbin, fvec = sherpa_stat.calc_stat(data[test_bin],
#                                                  model[test_bin],
#                                                  staterror=staterror[test_bin],
#                                                  extra_args=bkg_testbin)
#
#    actual_testbin = statsvec[test_bin]
#    #    assert_allclose(actual_testbin, desired_testbin)
#
#    # Now check total stat for all bins
#    desired, fvec = sherpa_stat.calc_stat(data, model, staterror=staterror,
#                                          extra_args=bkg)
#
#    actual = np.sum(statsvec)
#    print(fvec)
#    assert_allclose(actual, desired)
