# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from scipy.stats import ncx2, chi2
from scipy.optimize import brentq

from gammapy.utils.deprecation import deprecated_renamed_argument


__all__ = ["sigma_to_ts", "ts_to_sigma"]


@deprecated_renamed_argument("n_sigma_asimov", "asimov", "v2.0", arg_in_kwargs=True)
def sigma_to_ts(n_sigma, df=1, asimov=False):
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
    asimov : bool, optional
        Whether the number of sigma are calculated in an Asimov dataset.
        In that case the function applies the Wald test described in [2] and [3],
        where the TS of H1 under the H0 assumption is assumed to follow a non-central chi2 distribution.
        Should only be used for sensitivity computations. Default is False.

    Returns
    -------
    ts : float
        Test statistic value.

    References
    ----------
    .. [1] Wilks theorem: https://en.wikipedia.org/wiki/Wilks%27_theorem

    .. [2] Wald (1943): https://www.pp.rhul.ac.uk/~cowan/stat/wald1943.pdf

    .. [3] Cowan et al. (2011), European Physical Journal C, 71, 1554.
        doi:10.1140/epjc/s10052-011-1554-0.
    """
    # invert ts_to_sigma step
    p_value = chi2.sf(n_sigma**2, df=1)
    q_med = chi2.isf(p_value, df=df)

    # the simple case
    if not asimov or df == 1:
        return q_med

    def f(nc):
        return ncx2.ppf(0.5, df=df, nc=nc) - q_med

    # f rises monotonically with nc
    # if f(nc=0) is already positive there is no root to find
    if f(0.0) >= 0:
        return 0.0

    # solve for nc = ts
    ts = brentq(f, 0, q_med)

    return ts


@deprecated_renamed_argument("ts_asimov", "asimov", "v2.0", arg_in_kwargs=True)
def ts_to_sigma(ts, df=1, asimov=False):
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
    asimov : bool, optional
        Whether the number of sigma are calculated in an Asimov dataset.
        In that case the function applies the Wald test described in [2] and [3],
        where the TS of H1 under the H0 assumption is assumed to follow a non-central chi2 distribution.
        Should only be used for sensitivity computations. Default is False.


    Returns
    -------
    n_sigma : float
        Significance in number of sigma.

    References
    ----------
    .. [1] Wilks theorem: https://en.wikipedia.org/wiki/Wilks%27_theorem

    .. [2] Wald (1943): https://www.pp.rhul.ac.uk/~cowan/stat/wald1943.pdf

    .. [3] Cowan et al. (2011), European Physical Journal C, 71, 1554.
        doi:10.1140/epjc/s10052-011-1554-0.
    """
    q_med = ts
    if asimov and df != 1:
        q_med = ncx2.ppf(0.5, df=df, nc=ts)

    p_value = chi2.sf(q_med, df=df)
    return np.sqrt(chi2.isf(p_value, df=1))
