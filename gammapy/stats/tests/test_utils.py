# Licensed under a 3-clause BSD style license - see LICENSE.rst
from numpy.testing import assert_allclose
from gammapy.stats.utils import (
    sigma_to_ts,
    ts_to_sigma,
    sigma_to_ts_sensitivity,
    ts_to_sigma_sensitivity,
)


def test_sigma_ts_conversion():
    sigma_ref = 3
    ts_ref = 9
    df = 1
    ts = sigma_to_ts(3, df=df)
    assert_allclose(ts, ts_ref)
    ts = sigma_to_ts(3, df=df)
    assert_allclose(ts, ts_ref)
    sigma = ts_to_sigma(ts, df=df)
    assert_allclose(sigma, sigma_ref)
    sigma = ts_to_sigma(ts, df=df)
    assert_allclose(sigma, sigma_ref)

    df = 2
    ts_ref = 11.829158
    ts = sigma_to_ts(3, df=df)
    assert_allclose(ts, ts_ref)
    sigma = ts_to_sigma(ts, df=df)
    assert_allclose(sigma, sigma_ref)


def test_sigma_ts_sensitivity_conversion():
    sigma_ref = 3
    ts_ref = 9
    df = 1
    ts = sigma_to_ts_sensitivity(3, df=df)
    assert_allclose(ts, ts_ref)
    sigma = ts_to_sigma_sensitivity(ts, df=df)
    assert_allclose(sigma, sigma_ref)

    df = 10
    ts = sigma_to_ts_sensitivity(3, df=df)
    sigma = ts_to_sigma_sensitivity(ts, df=df)
    assert_allclose(sigma, sigma_ref)
