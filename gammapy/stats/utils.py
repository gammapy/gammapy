# Licensed under a 3-clause BSD style license - see LICENSE.rst

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
    return chi2.isf(p_value, df=df)


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
    return np.sqrt(chi2.isf(p_value, df=1))
