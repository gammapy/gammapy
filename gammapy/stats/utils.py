# Licensed under a 3-clause BSD style license - see LICENSE.rst

import mpmath
import numpy as np
from scipy.stats import chi2


def sigma_to_ts(n_sigma, df=1):
    """Convert number of sigma to delta ts according to the Wilks' theorem.

    The theorem is valid only if:
    - the two hypotheses tested can be defined in the same parameters space
    - the true value is not at the boundary of this parameters space.

    Parameters
    ----------
    n_sigma : float
        Significance in number of sigma.
    df : int
        Number of degree of freedom.

    Returns
    -------
    ts : float
        Test statistic

    Reference
    ---------
    Wilks theorem: https://en.wikipedia.org/wiki/Wilks%27_theorem
    """
    p_value = chi2.sf(n_sigma**2, df=1)
    ts = chi2.isf(p_value, df=df)

    invalid = np.atleast_1d(~np.isfinite(ts))
    if np.any(invalid):
        n_sigma = np.atleast_1d(n_sigma)
        sigma_invalid = n_sigma.astype(float)[invalid]
        df_invalid = df * np.ones(len(n_sigma))[invalid]

        ts_invalid = np.array(
            [
                _sigma_to_ts_mpmath(sig_val, df_val)
                for sig_val, df_val in zip(sigma_invalid, df_invalid)
            ]
        )
        try:
            ts[invalid] = ts_invalid
        except TypeError:
            ts = ts_invalid[0]
    return ts


def ts_to_sigma(ts, df=1):
    """Convert delta ts to number of sigma according to the Wilks' theorem.

    The theorem is valid only if :
    - the two hypotheses tested can be defined in the same parameters space
    - the true value is not at the boundary of this parameters space
    Reference:  https://en.wikipedia.org/wiki/Wilks%27_theorem

    Parameters
    ----------
    ts : float
        Test statistic.
    df : int
        Number of degree of freedom.

    Returns
    -------
    n_sigma : float
        Significance in number of sigma.
    """
    p_value = chi2.sf(ts, df=df)
    sigma = np.sqrt(chi2.isf(p_value, df=1))

    invalid = np.atleast_1d(~np.isfinite(sigma))
    if np.any(invalid):
        ts = np.atleast_1d(ts)
        ts_invalid = ts.astype(float)[invalid]
        df_invalid = df * np.ones(len(ts))[invalid]
        sigma_invalid = np.array(
            [
                _ts_to_sigma_mpmath(ts_val, df_val)
                for ts_val, df_val in zip(ts_invalid, df_invalid)
            ]
        )
        try:
            sigma[invalid] = sigma_invalid
        except TypeError:
            sigma = sigma_invalid[0]
    return sigma


def _chi2_cdf(x, df):
    x, df = mpmath.mpf(x), mpmath.mpf(df)
    return mpmath.gammainc(df / 2, 0, x / 2, regularized=True)


def _ts_to_sigma_mpmath(ts, df, rtol=1e-3, ndigit_min=1e3, ndigit_max=1e4):
    ndigit = np.maximum(100, ts / 4)
    ndigit = np.minimum(ndigit, 1e4)
    mpmath.mp.dps = int(ndigit)  # decimal digits of precision

    sigma_1df = np.sqrt(ts)
    sigma_range = np.linspace(0, sigma_1df, int(1.0 / rtol))[::-1]
    p = _chi2_cdf(ts, df=df)
    for sigma in sigma_range:
        if _chi2_cdf(sigma**2, df=1) <= p:
            break
    return sigma


def _sigma_to_ts_mpmath(
    sigma, df, rtol=1e-3, ndigit_min=1e3, ndigit_max=1e4, ts_max=1e6
):
    ts_1df = sigma**2.0

    ndigit = np.maximum(100, ts_1df / 4)
    ndigit = np.minimum(ndigit, 1e4)
    mpmath.mp.dps = int(ndigit)  # decimal digits of precision

    ts_range = np.arange(ts_1df, ts_max, ts_1df * rtol)
    p = _chi2_cdf(sigma**2, df=1)
    for ts in ts_range:
        if _chi2_cdf(ts, df=df) >= p:
            break
    return ts
