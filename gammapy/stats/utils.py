# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from scipy.stats import ncx2, chi2
from scipy.optimize import brentq
from gammapy.utils.deprecation import deprecated_renamed_argument
from gammapy.utils.roots import find_roots

__all__ = ["sigma_to_ts", "ts_to_sigma"]


@deprecated_renamed_argument("n_sigma_asimov", None, "2.2")
def sigma_to_ts(n_sigma, df=1, n_sigma_asimov=0):
    """Convert number of sigma to delta ts.

    Assumes that the TS follows a chi2 distribution according to Wilks theorem [1].
    This is valid only if:

    - the two hypotheses tested can be defined in the same parameters space
    - the true value is not at the boundary of this parameters space.

    Parameters
    ----------
    n_sigma : float
        Significance in number of sigma.
    df : int, optional
        Number of degree of freedom. Default is 1.
    n_sigma_asimov : float, optional
        Significance in number of sigma in the Asimov dataset.
        Should only be used for sensitivity computations. Default is 0.
        Deprecated since v2.2.
    Returns
    -------
    ts : float
        Test statistic value.

    References
    ----------
    .. [1] Wilks theorem: https://en.wikipedia.org/wiki/Wilks%27_theorem
    """
    # invert ts_to_sigma step
    p_value = chi2.sf(n_sigma**2, df=1)
    ts = chi2.isf(p_value, df=df)

    return ts


def sigma_to_ts_sensitivity(n_sigma, df=1):
    """Convert median significance to an Asimov test statistic.

    Under the regular Wald approximation [1]_ [2]_, the test statistic
    follows a non-central chi-square distribution under the alternative
    hypothesis. Its noncentrality parameter is estimated by the test
    statistic evaluated on an Asimov dataset. The null distribution is
    a central chi-square distribution with the same degrees of freedom.

    Parameters
    ----------
    n_sigma : float
        Target median significance in non-negative Gaussian sigma, using
        the two-sided convention ``p = 2 * norm.sf(n_sigma)``.
    df : int, optional
        Number of degree of freedom. Default is 1.

    Returns
    -------
    ts : float
        Asimov test statistic, interpreted as the noncentrality parameter.
        Returns zero for targets below the minimum attainable median
        significance.

    References
    ----------
    .. [1] Wald (1943): https://www.pp.rhul.ac.uk/~cowan/stat/wald1943.pdf

    .. [2] Cowan et al. (2011), European Physical Journal C, 71, 1554.
        doi:10.1140/epjc/s10052-011-1554-0.
    """
    # invert ts_to_sigma step
    p_value = chi2.sf(n_sigma**2, df=1)
    q_med = chi2.isf(p_value, df=df)

    def f(nc):
        return ncx2.ppf(0.5, df=df, nc=nc) - q_med

    # f rises monotonically with nc
    # if f(nc=0) is already positive there is no root to find
    if f(0.0) >= 0:
        return 0.0

    # solve for nc = ts
    roots, _ = find_roots(f, 0, q_med)
    return roots[0]


@deprecated_renamed_argument("ts_asimov", None, "2.2")
def ts_to_sigma(ts, df=1, ts_asimov=0):
    """Convert delta ts to number of sigma.

    Assumes that the TS follows a chi2 distribution according to Wilks theorem [1].
    This is valid only if:

    - the two hypotheses tested can be defined in the same parameters space
    - the true value is not at the boundary of this parameters space.

    Parameters
    ----------
    ts : float
        Test statistic value.
    df : int, optional
        Number of degree of freedom. Default is 1.
    ts_asimov : float, optional
        TS value in the Asimov dataset.
        Should only be used for sensitivity computations. Default is 0.
        Deprecated since v2.2.
    Returns
    -------
    n_sigma : float
        Significance in number of sigma.

    References
    ----------
    .. [1] Wilks theorem: https://en.wikipedia.org/wiki/Wilks%27_theorem
    """
    p_value = chi2.sf(ts, df=df)
    return np.sqrt(chi2.isf(p_value, df=1))


def ts_to_sigma_sensitivity(ts, df=1):
    """Convert an Asimov test statistic to median significance.

    Under the regular Wald approximation [1]_ [2]_, the test statistic
    follows a non-central chi-square distribution under the alternative
    hypothesis. Its noncentrality parameter is estimated by the test
    statistic evaluated on an Asimov dataset. The null distribution is
    a central chi-square distribution with the same degrees of freedom.

    Parameters
    ----------
    ts : float
        Non-negative Asimov test statistic, interpreted as the
        noncentrality parameter. Only use for sensitivity computations.
    df : int, optional
        Number of degree of freedom. Default is 1.

    Returns
    -------
    n_sigma : float
        Median significance in Gaussian sigma, using the two-sided
        convention ``n_sigma = norm.isf(p / 2)``.

    References
    ----------
    .. [1] Wald (1943): https://www.pp.rhul.ac.uk/~cowan/stat/wald1943.pdf

    .. [2] Cowan et al. (2011), European Physical Journal C, 71, 1554.
        doi:10.1140/epjc/s10052-011-1554-0.
    """
    q_med = ncx2.ppf(0.5, df=df, nc=ts)

    p_value = chi2.sf(q_med, df=df)
    return np.sqrt(chi2.isf(p_value, df=1))
