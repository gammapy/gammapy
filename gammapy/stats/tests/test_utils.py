# Licensed under a 3-clause BSD style license - see LICENSE.rst
from numpy.testing import assert_allclose
from gammapy.stats.utils import sigma_to_ts, ts_to_sigma


def test_sigma_ts_conversion():

    sigma_ref = 3
    df = 1
    ts = sigma_to_ts(3, df=df)
    assert_allclose(ts, sigma_ref**2.0)
    sigma = ts_to_sigma(ts, df=df)
    assert_allclose(sigma, sigma_ref)

    df = 2
    ts = sigma_to_ts(3, df=df)
    sigma = ts_to_sigma(ts, df=df)
    assert_allclose(sigma, sigma_ref)
