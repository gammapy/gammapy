# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Conversion functions for test statistic <-> significance <-> probability."""
import numpy as np
from scipy.stats import norm
from scipy.special import erfinv, erf
from scipy.optimize import fsolve


__all__ = [
    "significance_to_probability_normal",
    "probability_to_significance_normal",
    "probability_to_significance_normal_limit",
    "significance_to_probability_normal_limit",
]


def significance_to_probability_normal(significance):
    """Convert significance to one-sided tail probability.

    Parameters
    ----------
    significance : array_like
        Significance

    Returns
    -------
    probability : `numpy.ndarray`
        One-sided tail probability

    See Also
    --------
    probability_to_significance_normal,
    significance_to_probability_normal_limit

    Examples
    --------
    >>> significance_to_probability_normal(0)
    0.5
    >>> significance_to_probability_normal(1)
    0.15865525393145707
    >>> significance_to_probability_normal(3)
    0.0013498980316300933
    >>> significance_to_probability_normal(5)
    2.8665157187919328e-07
    >>> significance_to_probability_normal(10)
    7.6198530241604696e-24
    """
    return norm.sf(significance)


def probability_to_significance_normal(probability):
    """Convert one-sided tail probability to significance.

    Parameters
    ----------
    probability : array_like
        One-sided tail probability

    Returns
    -------
    significance : ndarray
        Significance

    See Also
    --------
    significance_to_probability_normal,
    probability_to_significance_normal_limit

    Examples
    --------
    >>> probability_to_significance_normal(1e-10)
    6.3613409024040557
    """
    return norm.isf(probability)


def _p_to_s_direct(probability, one_sided=True):
    """Direct implementation of p_to_s for checking.

    Reference: RooStats User Guide Equations (6,7).
    """
    probability = 1 - probability  # We want p to be the tail probability
    temp = np.where(one_sided, 2 * probability - 1, probability)
    return np.sqrt(2) * erfinv(temp)


def _s_to_p_direct(significance, one_sided=True):
    """Direct implementation of s_to_p for checking.

    Note: _p_to_s_direct was solved for p.
    """
    temp = erf(significance / np.sqrt(2))
    probability = np.where(one_sided, (temp + 1) / 2.0, temp)
    return 1 - probability  # We want p to be the tail probability


def probability_to_significance_normal_limit(probability):
    r"""Tail probability to significance in small probability limit.

    Reference: https://ui.adsabs.harvard.edu/abs/2007physics...2156C, Equation (4)

    They say it is better than 1% for s > 1.6.

    Asymptotically: :math:`s \sim \sqrt{-\log(p)}`

    See also: `significance_to_probability_normal_limit`
    """
    u = -2 * np.log(probability * np.sqrt(2 * np.pi))
    return np.sqrt(u - np.log(u))


def significance_to_probability_normal_limit(significance, guess=1e-100):
    """Significance to tail probability in large significance limit.

    See also: `probability_to_significance_normal_limit`
    """
    # Note: s^2 = u - log(u) can't be solved analytically.
    def f(probability):
        if probability > 0:
            return probability_to_significance_normal_limit(probability) - significance
        else:
            return 1e100

    return fsolve(f, guess)
