# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Miscellaneous utility functions
"""
from __future__ import print_function, division
import numpy as np


__all__ = ['s_to_p', 'p_to_s', 'p_to_s_limit', 's_to_p_limit', 'cov_to_corr']

__doctest_skip__ = ['*']


def s_to_p(s):
    """Convert significance to one-sided tail probability.

    Parameters
    ----------
    s : array_like
        Significance

    Returns
    -------
    p : ndarray
        One-sided tail probability

    See Also
    --------
    p_to_s, s_to_p_limit

    Examples
    --------
    >>> s_to_p(0)
    0.5
    >>> s_to_p(1)
    0.15865525393145707
    >>> s_to_p(3)
    0.0013498980316300933
    >>> s_to_p(5)
    2.8665157187919328e-07
    >>> s_to_p(10)
    7.6198530241604696e-24
    """
    from scipy.stats import norm
    return norm.sf(s)


def p_to_s(p):
    """Convert one-sided tail probability to significance.

    Parameters
    ----------
    p : array_like
        One-sided tail probability

    Returns
    -------
    s : ndarray
        Significance

    See Also
    --------
    s_to_p, p_to_s_limit

    Examples
    --------
    >>> p_to_s(1e-10)
    6.3613409024040557
    """
    from scipy.stats import norm
    return norm.isf(p)


def _p_to_s_direct(p, one_sided=True):
    """Direct implementation of p_to_s for checking.

    Reference: RooStats User Guide Equations (6,7).
    """
    from scipy.special import erfinv
    p = 1 - p  # We want p to be the tail probability
    temp = np.where(one_sided, 2 * p - 1, p)
    return np.sqrt(2) * erfinv(temp)


def _s_to_p_direct(s, one_sided=True):
    """Direct implementation of s_to_p for checking.

    @see: _p_to_s_direct was solved for p.
    """
    from scipy.special import erf
    temp = erf(s / np.sqrt(2))
    p = np.where(one_sided, (temp + 1) / 2., temp)
    return 1 - p  # We want p to be the tail probability


def p_to_s_limit(p):
    """Convert tail probability to significance
    in the limit of small p and large s.

    Reference: Equation (4) of
    http://adsabs.harvard.edu/abs/2007physics...2156C
    They say it is better than 1% for s > 1.6.

    Asymptotically: s ~ sqrt(-log(p))
    """
    u = -2 * np.log(p * np.sqrt(2 * nppi))
    return np.sqrt(u - np.log(u))


def s_to_p_limit(s, guess=1e-100):
    """Convert significance to tail probability
    in the limit of small p and large s.

    @see: p_to_s_limit docstring
    @note s^2 = u - log(u) can't be solved analytically.
    """
    from scipy.optimize import fsolve

    def f(p):
        return p_to_s_limit(p) - s if p > 0 else 1e100

    return fsolve(f, guess)


def cov_to_corr(covariance):
    """
    Compute correlation matrix from covariance matrix.

    The correlation matrix :math:`c` is related to the covariance matrix :math:`\\sigma` by:

    .. math::

        c_{ij} = \\frac{\\sigma_{ij}}{\\sqrt{\\sigma_{ii} \sigma_{jj}}}

    Parameters
    ----------
    covariance : `numpy.array`
        Covariance matrix.

    Returns
    -------
    correlation : `numpy.array`
        Correlation matrix.
    """
    diagonal = np.sqrt(covariance.diagonal())
    return (covariance.T / diagonal).T / diagonal
