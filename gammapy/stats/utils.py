# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from scipy.stats import ncx2


def sigma_to_ts(n_sigma, df=1, n_sigma_asymov=0):
    """Convert number of sigma to delta ts.

    Applies Wilks theorem [1]. This is valid only if:
    - the two hypotheses tested can be defined in the same parameters space
    - the true value is not at the boundary of this parameters space.

    Parameters
    ----------
    n_sigma : float
        Significance in number of sigma.
    df : int
        Number of degree of freedom.
    n_sigma_asymov : float
        Significance in number of sigma in the Asymov dataset
        (in which counts equal to the predicted counts).
        In that case it applies the Wald test described in [2] and [3].
        Should be used only for sensitivity computations.

    Returns
    -------
    ts : float
        Test statistic

    References
    ----------
    .. [1] Wilks theorem: https://en.wikipedia.org/wiki/Wilks%27_theorem

    .. [2] Wald (1943): https://www.pp.rhul.ac.uk/~cowan/stat/wald1943.pdf

    .. [3] Cowan et al. (2011), European Physical Journal C, 71, 1554.
        doi:10.1140/epjc/s10052-011-1554-0.
    """
    ts_asymov = n_sigma_asymov**2
    p_value = ncx2.sf(n_sigma**2, df=1, nc=ts_asymov)
    return ncx2.isf(p_value, df=df, nc=ts_asymov)


def ts_to_sigma(ts, df=1, ts_asymov=0):
    """Convert delta ts to number of sigma.

    Applies Wilks theorem [1]. This is valid only if:
    - the two hypotheses tested can be defined in the same parameters space
    - the true value is not at the boundary of this parameters space.

    Parameters
    ----------
    ts : float
        Test statistic.
    df : int
        Number of degree of freedom.
    ts_asymov : float
        TS value in the Asymov dataset
        (in which counts equal to the predicted counts).
        In that case it applies the Wald test described in [2] and [3].
        Should be used only for sensitivity computations.


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
    p_value = ncx2.sf(ts, df=df, nc=ts_asymov)
    return np.sqrt(ncx2.isf(p_value, df=1, nc=ts_asymov))
