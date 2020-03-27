# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Poisson significance computations for these two cases.

* known background level ``mu_bkg``
* background estimated from ``n_off`
"""
import numpy as np
import scipy.optimize
import scipy.special
import scipy.stats
from .normal import significance_to_probability_normal

__all__ = [
    "excess_matching_significance",
    "excess_matching_significance_on_off",
    "excess_ul_helene",
]

__doctest_skip__ = ["*"]



def excess_ul_helene(excess, excess_error, significance):
    """Compute excess upper limit using the Helene method.

    Reference: https://ui.adsabs.harvard.edu/abs/1984NIMPA.228..120H

    Parameters
    ----------
    excess : float
        Signal excess
    excess_error : float
        Gaussian excess error
        For on / off measurement, use this function to compute it:
        `~gammapy.stats.excess_error`.
    significance : float
        Confidence level significance for the excess upper limit.

    Returns
    -------
    excess_ul : float
        Upper limit for the excess
    """
    conf_level1 = significance_to_probability_normal(significance)

    if excess_error <= 0:
        raise ValueError(f"Non-positive excess_error: {excess_error}")

    if excess >= 0.0:
        zeta = excess / excess_error
        value = zeta / np.sqrt(2.0)
        integral = (1.0 + scipy.special.erf(value)) / 2.0
        integral2 = 1.0 - conf_level1 * integral
        value_old = value
        value_new = value_old + 0.01
        if integral > integral2:
            value_new = 0.0
        integral = (1.0 + scipy.special.erf(value_new)) / 2.0
    else:
        zeta = -excess / excess_error
        value = zeta / np.sqrt(2.0)
        integral = 1 - (1.0 + scipy.special.erf(value)) / 2.0
        integral2 = 1.0 - conf_level1 * integral
        value_old = value
        value_new = value_old + 0.01
        integral = (1.0 + scipy.special.erf(value_new)) / 2.0

    # The 1st Loop is for Speed & 2nd For Precision
    while integral < integral2:
        value_old = value_new
        value_new = value_new + 0.01
        integral = (1.0 + scipy.special.erf(value_new)) / 2.0
    value_new = value_old + 0.0000001
    integral = (1.0 + scipy.special.erf(value_new)) / 2.0

    while integral < integral2:
        value_new = value_new + 0.0000001
        integral = (1.0 + scipy.special.erf(value_new)) / 2.0
    value_new = value_new * np.sqrt(2.0)

    if excess >= 0.0:
        conf_limit = (value_new + zeta) * excess_error
    else:
        conf_limit = (value_new - zeta) * excess_error

    return conf_limit


def excess_matching_significance(mu_bkg, significance, method="lima"):
    r"""Compute excess matching a given significance.

    This function is the inverse of `significance`.

    Parameters
    ----------
    mu_bkg : array_like
        Known background level
    significance : array_like
        Significance
    method : {'lima', 'simple'}
        Select method

    Returns
    -------
    excess : `numpy.ndarray`
        Excess

    See Also
    --------
    significance, excess_matching_significance_on_off

    Examples
    --------
    >>> excess_matching_significance(mu_bkg=0.2, significance=5, method='lima')
    TODO
    >>> excess_matching_significance(mu_bkg=0.2, significance=5, method='simple')
    TODO
    """
    mu_bkg = np.asanyarray(mu_bkg, dtype=np.float64)
    significance = np.asanyarray(significance, dtype=np.float64)

    if method == "simple":
        return _excess_matching_significance_simple(mu_bkg, significance)
    elif method == "lima":
        return _excess_matching_significance_lima(mu_bkg, significance)
    else:
        raise ValueError(f"Invalid method: {method}")


def excess_matching_significance_on_off(n_off, alpha, significance, method="lima"):
    r"""Compute sensitivity of an on-off observation.

    This function is the inverse of `significance_on_off`.

    Parameters
    ----------
    n_off : array_like
        Observed number of counts in the off region
    alpha : array_like
        On / off region exposure ratio for background events
    significance : array_like
        Desired significance level
    method : {'lima', 'simple'}
        Which method?

    Returns
    -------
    excess : `numpy.ndarray`
        Excess

    See Also
    --------
    significance_on_off, excess_matching_significance

    Examples
    --------
    >>> excess_matching_significance_on_off(n_off=20,alpha=0.1,significance=5,method='lima')
    12.038
    >>> excess_matching_significance_on_off(n_off=20,alpha=0.1,significance=5,method='simple')
    27.034
    >>> excess_matching_significance_on_off(n_off=20,alpha=0.1,significance=0,method='lima')
    2.307301461e-09
    >>> excess_matching_significance_on_off(n_off=20,alpha=0.1,significance=0,method='simple')
    0.0
    >>> excess_matching_significance_on_off(n_off=20,alpha=0.1,significance=-10,method='lima')
    nan
    >>> excess_matching_significance_on_off(n_off=20,alpha=0.1,significance=-10,method='simple')
    nan
    """
    n_off = np.asanyarray(n_off, dtype=np.float64)
    alpha = np.asanyarray(alpha, dtype=np.float64)
    significance = np.asanyarray(significance, dtype=np.float64)

    if method == "simple":
        return _excess_matching_significance_on_off_simple(n_off, alpha, significance)
    elif method == "lima":
        return _excess_matching_significance_on_off_lima(n_off, alpha, significance)
    else:
        raise ValueError(f"Invalid method: {method}")


def _excess_matching_significance_simple(mu_bkg, significance):
    return significance * np.sqrt(mu_bkg)


def _excess_matching_significance_on_off_simple(n_off, alpha, significance):
    # TODO: can these equations be simplified?
    significance2 = significance ** 2
    determinant = significance2 + 4 * n_off * alpha * (1 + alpha)
    temp = significance2 + 2 * n_off * alpha
    n_on = 0.5 * (temp + significance * np.sqrt(np.abs(determinant)))
    return n_on - background(n_off, alpha)


# This is mostly a copy & paste from _excess_matching_significance_on_off_lima
# TODO: simplify this, or avoid code duplication?
# Looking at the formula for significance_lima_on_off, I don't think
# it can be analytically inverted because the n_on appears inside and outside the log
# So probably root finding is still needed here.
def _excess_matching_significance_lima(mu_bkg, significance):
    # Significance not well-defined for n_on < 0
    # Return Nan if given significance can't be reached
    s0 = _significance_lima(n_on=1e-5, mu_bkg=mu_bkg)
    if s0 >= significance:
        return np.nan

    def target_significance(n_on):
        if n_on >= 0:
            return _significance_lima(n_on, mu_bkg) - significance
        else:
            # This high value is to tell the optimizer to stay n_on >= 0
            return 1e10

    excess_guess = _excess_matching_significance_simple(mu_bkg, significance)
    n_on_guess = excess_guess + mu_bkg

    # solver options to control robustness / accuracy / speed
    opts = dict(factor=0.1)
    n_on = scipy.optimize.fsolve(target_significance, n_on_guess, **opts)
    return n_on - mu_bkg


def _excess_matching_significance_on_off_lima(n_off, alpha, significance):
    # Significance not well-defined for n_on < 0
    # Return Nan if given significance can't be reached
    s0 = _significance_lima_on_off(n_on=1e-5, n_off=n_off, alpha=alpha)
    if s0 >= significance:
        return np.nan

    def target_significance(n_on):
        if n_on >= 0:
            return _significance_lima_on_off(n_on, n_off, alpha) - significance
        else:
            # This high value is to tell the optimizer to stay n_on >= 0
            return 1e10

    excess_guess = _excess_matching_significance_on_off_simple(
        n_off, alpha, significance
    )
    n_on_guess = excess_guess + background(n_off, alpha)

    # solver options to control robustness / accuracy / speed
    opts = dict(factor=0.1)
    n_on = scipy.optimize.fsolve(target_significance, n_on_guess, **opts)
    return n_on - background(n_off, alpha)


_excess_matching_significance_lima = np.vectorize(_excess_matching_significance_lima)
_excess_matching_significance_on_off_lima = np.vectorize(
    _excess_matching_significance_on_off_lima
)
