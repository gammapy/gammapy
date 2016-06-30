# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from ... import stats as gammapy_stats
from ...utils.testing import requires_dependency
from ...utils.random import get_random_state

def get_test_data():
    random_state = get_random_state(3)
    # put factor to 100 to not run into special cases in WStat
    model = random_state.rand(10) * 10
    data = random_state.poisson(model)
    staterror = np.sqrt(data)
    off_vec = random_state.poisson(0.7 * model)
    return data, model, staterror, off_vec


@requires_dependency('sherpa')
def test_cstat():
    import sherpa.stats as ss
    sherpa_stat = ss.CStat()
    data, model, staterror, off_vec = get_test_data()
    desired, fvec  = sherpa_stat.calc_stat(data, model, staterror=staterror)
        
    statsvec = gammapy_stats.cstat(n_observed=data, mu_predicted=model)
    actual = np.sum(statsvec)
    assert_allclose(actual, desired) 


@requires_dependency('sherpa')
def test_cash():
    import sherpa.stats as ss
    sherpa_stat = ss.Cash()
    data, model, staterror, off_vec = get_test_data()
    desired, fvec  = sherpa_stat.calc_stat(data, model, staterror=staterror)
        
    statsvec = gammapy_stats.cash(n_observed=data, mu_predicted=model)
    actual = np.sum(statsvec)
    assert_allclose(actual, desired) 


@pytest.mark.xfail(reason='values do not match, under investigation')
@requires_dependency('sherpa')
def test_wstat():
    import sherpa.stats as ss
    sherpa_stat = ss.WStat()
    data, model, staterror, off_vec = get_test_data()
    alpha = 0.2
    bkg_vec = alpha * off_vec
    # This is how sherpa wants the background (found by trial and error)
    bkg = dict(bkg=off_vec,
               exposure_time=[1, 1],
               backscale_ratio=[1./alpha] * len(data),
               data_size=len(data)
              )
    desired, fvec  = sherpa_stat.calc_stat(data, model, staterror=staterror,
                                           bkg=bkg)
        
    statsvec = gammapy_stats.wstat(n_on=data, mu_signal=model, n_bkg=bkg_vec)
    actual = np.sum(statsvec)
    assert_allclose(actual, desired) 

