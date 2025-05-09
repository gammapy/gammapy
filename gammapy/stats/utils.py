# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from scipy.stats import ncx2


__all__ = ["sigma_to_ts", "ts_to_sigma"]


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
        Significance in number of sigma in the Asimov dataset
        (in which counts are equal to the predicted counts).
        In that case the function applies the Wald test described in [2] and [3],
        where the TS of H1 under the H0 assumption is assumed to follow a non-central chi2 distribution.
        Should only be used for sensitivity computations. Default is 0.

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
    ts_asimov = n_sigma_asimov**2
    p_value = ncx2.sf(n_sigma**2, df=1, nc=ts_asimov)
    return ncx2.isf(p_value, df=df, nc=ts_asimov)


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
        TS value in the Asimov dataset
        (in which counts equal to the predicted counts).
        In that case the function applies the Wald test described in [2] and [3],
        and the TS is assumed to follow a non-central chi2 distribution.
        Should only be used for sensitivity computations. Default is 0.


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
    p_value = ncx2.sf(ts, df=df, nc=ts_asimov)
    return np.sqrt(ncx2.isf(p_value, df=1, nc=ts_asimov))
