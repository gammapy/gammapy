# Licensed under a 3-clause BSD style license - see LICENSE.rst
from numpy.testing import assert_allclose
from gammapy.stats.utils import sigma_to_ts, ts_to_sigma


def test_sigma_ts_conversion():
    sigma_ref = 3
    ts_ref = 9
    df = 1
    ts = sigma_to_ts(3, df=df)
    assert_allclose(ts, ts_ref)
    sigma = ts_to_sigma(ts, df=df)
    assert_allclose(sigma, sigma_ref)

    df = 2
    ts_ref = 11.829158
    ts = sigma_to_ts(3, df=df)
    assert_allclose(ts, ts_ref)
    sigma = ts_to_sigma(ts, df=df)
    assert_allclose(sigma, sigma_ref)

    df = 1
    sigma_ref = 50
    ts_ref = 2500
    ts = sigma_to_ts(sigma_ref, df=df)
    assert_allclose(ts, ts_ref)
    sigma = ts_to_sigma(ts, df=df)
    assert_allclose(sigma, sigma_ref)

    df = 10
    sigma_ref = 50
    ts_ref = 2560
    ts = sigma_to_ts(sigma_ref, df=df)
    assert_allclose(ts, ts_ref, rtol=1e-3)
    sigma = ts_to_sigma(ts, df=df)
    assert_allclose(sigma, sigma_ref, rtol=1e-3)
