"""
Implementation of the Sherpa statistics with numpy

All functions return arrays::

    >>> stat_per_bin = stat(...)
    If you want the sum say so:
    >>> total_stat = stat(...).sum()

WARNING: I just typed this, no documentation or testing yet ...

Todo
----

- check corner cases (e.g. counts or exposure zero)
- exception handling for bad input?

References
----------

* http://cxc.cfa.harvard.edu/sherpa/statistics
* https://github.com/taldcroft/sherpa/blob/master/stats/__init__.py
* sherpa/include/sherpa/stats.hh contains the C++ implementations of the Sherpa stats

"""

import numpy as np
from numpy import log, sqrt

__all__ = ('cash', 'cstat', 'chi2constvar', 'chi2datavar',
           'chi2gehrels', 'chi2modvar', 'chi2xspecvar')


def cash(D, M):
    """Cash statistic
    """
    D = np.asanyarray(D, dtype=np.float64)
    M = np.asanyarray(M, dtype=np.float64)

    stat = 2 * (M - D * log(M))
    stat = np.where(M > 0, stat, 0)

    return stat


def cstat(D, M):
    """C statistic
    """
    D = np.asanyarray(D, dtype=np.float64)
    M = np.asanyarray(M, dtype=np.float64)

    stat = 2 * (M - D + D * (log(D) - log(M)))
    stat = np.where(M > 0, stat, 0)
    # TODO: handle bins where D = 0 (can't call log!)
    # Check what Sherpa does in those cases ...

    return stat


def chi2(N_S, B, S, sigma2):
    """Chi ^ 2 statistic
    """
    N_S = np.asanyarray(N_S, dtype=np.float64)
    B = np.asanyarray(B, dtype=np.float64)
    S = np.asanyarray(S, dtype=np.float64)
    sigma2 = np.asanyarray(sigma2, dtype=np.float64)

    return (N_S - B - S) ** 2 / sigma2


def chi2constvar(N_S, N_B, A_S, A_B):
    """Chi ^ 2 with constant variance
    """
    N_S = np.asanyarray(N_S, dtype=np.float64)
    N_B = np.asanyarray(N_B, dtype=np.float64)
    A_S = np.asanyarray(A_S, dtype=np.float64)
    A_B = np.asanyarray(A_B, dtype=np.float64)

    alpha2 = (A_S / A_B) ** 2
    # Need to mulitply with np.ones_like(N_S) here?
    sigma2 = (N_S + alpha2 * N_B).mean()
    return chi2(N_S, A_B, A_S, sigma2)


def chi2datavar(N_S, N_B, A_S, A_B):
    """Chi ^ 2 with data variance
    """
    N_S = np.asanyarray(N_S, dtype=np.float64)
    N_B = np.asanyarray(N_B, dtype=np.float64)
    A_S = np.asanyarray(A_S, dtype=np.float64)
    A_B = np.asanyarray(A_B, dtype=np.float64)
    alpha2 = (A_S / A_B) ** 2
    sigma2 = N_S + alpha2 * N_B
    return chi2(N_S, A_B, A_S, sigma2)


def chi2gehrels(N_S, N_B, A_S, A_B):
    """Chi ^ 2 with Gehrel's variance
    """
    N_S = np.asanyarray(N_S, dtype=np.float64)
    N_B = np.asanyarray(N_B, dtype=np.float64)
    A_S = np.asanyarray(A_S, dtype=np.float64)
    A_B = np.asanyarray(A_B, dtype=np.float64)
    alpha2 = (A_S / A_B) ** 2
    sigma_S = 1 + sqrt(N_S + 0.75)
    sigma_B = 1 + sqrt(N_B + 0.75)
    sigma2 = sigma_S ** 2 + alpha2 * sigma_B ** 2
    return chi2(N_S, A_B, A_S, sigma2)


def chi2modvar(S, B, A_S, A_B):
    """Chi ^ 2 with model variance
    """
    return chi2datavar(S, B, A_S, A_B)


def chi2xspecvar(N_S, N_B, A_S, A_B):
    """Chi ^ 2 with XSPEC variance
    """
    # TODO: is this correct?
    mask = (N_S < 1) | (N_B < 1)
    # _stat = np.empty_like(mask, dtype='float')
    # _stat[mask] = 1
    return np.where(mask, 1, chi2datavar(N_S, N_B, A_S, A_B))
