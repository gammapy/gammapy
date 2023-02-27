# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from scipy.stats import chi2


def sigma_to_ts(n_sigma, df=1):
    """Convert number of sigma to delta ts."""
    p_value = chi2.sf(n_sigma**2, df=1)
    return chi2.isf(p_value, df=df)


def ts_to_sigma(ts, df=1):
    """Convert delta ts to number of sigma"""
    p_value = chi2.sf(ts, df=df)
    return np.sqrt(chi2.isf(p_value, df=1))
