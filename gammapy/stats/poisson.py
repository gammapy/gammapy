# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Poisson significance computations for these two cases.

* known background level ``mu_bkg``
* background estimated from ``n_off`
"""
import numpy as np
from scipy.stats import norm, poisson
from scipy.special import erf
from scipy.optimize import fsolve
from .normal import significance_to_probability_normal

__all__ = [
    "background",
    "background_error",
    "excess",
    "excess_error",
    "significance",
    "significance_on_off",
    "excess_matching_significance",
    "excess_matching_significance_on_off",
    "excess_ul_helene",
]

__doctest_skip__ = ["*"]


def background(n_off, alpha):
    r"""Estimate background in the on-region from an off-region observation.

    .. math::
        \mu_{background} = \alpha \times n_{off}

    Parameters
    ----------
    n_off : array_like
        Observed number of counts in the off region
    alpha : array_like
        On / off region exposure ratio for background events

    Returns
    -------
    background : `numpy.ndarray`
        Background estimate for the on region

    Examples
    --------
    >>> background(n_off=4, alpha=0.1)
    0.4
    >>> background(n_off=9, alpha=0.2)
    1.8
    """
    n_off = np.asanyarray(n_off, dtype=np.float64)
    alpha = np.asanyarray(alpha, dtype=np.float64)

    return alpha * n_off


def background_error(n_off, alpha):
    r"""Estimate standard error on background in the on region from an off-region observation.

    .. math::
          \Delta\mu_{bkg} = \alpha \times \sqrt{n_{off}}

    Parameters
    ----------
    n_off : array_like
        Observed number of counts in the off region
    alpha : array_like
        On / off region exposure ratio for background events

    Returns
    -------
    background : `numpy.ndarray`
        Background estimate for the on region

    Examples
    --------
    >>> background_error(n_off=4, alpha=0.1)
    0.2
    >>> background_error(n_off=9, alpha=0.2)
    0.6
    """
    n_off = np.asanyarray(n_off, dtype=np.float64)
    alpha = np.asanyarray(alpha, dtype=np.float64)

    return alpha * np.sqrt(n_off)


def excess(n_on, n_off, alpha):
    r"""Estimate excess in the on region for an on-off observation.

    .. math::
          \mu_{excess} = n_{on} - \alpha \times n_{off}

    Parameters
    ----------
    n_on : array_like
        Observed number of counts in the on region
    n_off : array_like
        Observed number of counts in the off region
    alpha : array_like
        On / off region exposure ratio for background events

    Returns
    -------
    excess : `numpy.ndarray`
        Excess estimate for the on region

    Examples
    --------
    >>> excess(n_on=10, n_off=20, alpha=0.1)
    8.0
    >>> excess(n_on=4, n_off=9, alpha=0.5)
    -0.5
    """
    n_on = np.asanyarray(n_on, dtype=np.float64)
    n_off = np.asanyarray(n_off, dtype=np.float64)
    alpha = np.asanyarray(alpha, dtype=np.float64)

    return n_on - alpha * n_off


def excess_error(n_on, n_off, alpha):
    r"""Estimate error on excess for an on-off measurement.

    .. math::
        \Delta\mu_{excess} = \sqrt{n_{on} + \alpha ^ 2 \times n_{off}}

    TODO: Implement better error and limit estimates (Li & Ma, Rolke)!

    Parameters
    ----------
    n_on : array_like
        Observed number of counts in the on region
    n_off : array_like
        Observed number of counts in the off region
    alpha : array_like
        On / off region exposure ratio for background events

    Returns
    -------
    excess_error : `numpy.ndarray`
        Excess error estimate

    Examples
    --------
    >>> excess_error(n_on=10, n_off=20, alpha=0.1)
    3.1937438845342623...
    >>> excess_error(n_on=4, n_off=9, alpha=0.5)
    2.5
    """
    n_on = np.asanyarray(n_on, dtype=np.float64)
    n_off = np.asanyarray(n_off, dtype=np.float64)
    alpha = np.asanyarray(alpha, dtype=np.float64)

    variance = n_on + (alpha ** 2) * n_off
    return np.sqrt(variance)


def significance(n_on, mu_bkg, method="lima", n_on_min=1):
    r"""Compute significance for an observed number of counts and known background.

    The default ``method="lima"`` gives the significance estimate corresponding
    to equation (17) from the Li & Ma paper [1]_ in the limiting of known background
    :math:`\mu_{bkg} = \alpha \times n_{off}` with :math:`\alpha \to 0`.

    It is given by the following formula:

    .. math::
        S_{lima} = \left[
          2 n_{on} \log \left( \frac{n_{on}}{\mu_{bkg}} \right) - n_{on} + \mu_{bkg}
        \right] ^ {1/2}

    For ``method="simple"``, the significance estimate is given by:

    .. math::
        S_{simple} = (n_{on} - \mu_{bkg}) / \sqrt{\mu_{bkg}}


    Parameters
    ----------
    n_on : array_like
        Observed number of counts
    mu_bkg : array_like
        Known background level
    method : {"lima", "simple"}
        Method for significance estimation
    n_on_min : float
        Minimum ``n_on`` (return ``NaN`` for smaller values)

    Returns
    -------
    significance : `~numpy.ndarray`
        Significance estimate

    References
    ----------
    .. [1] Li and Ma, "Analysis methods for results in gamma-ray astronomy",
       `Link <https://ui.adsabs.harvard.edu/abs/1983ApJ...272..317L>`_

    See Also
    --------
    excess, significance_on_off

    Examples
    --------
    >>> significance(n_on=10, mu_bkg=2, method='lima')
    4.0235256
    >>> significance(n_on=10, mu_bkg=2, method='simple')
    5.65685425
    """
    n_on = np.asanyarray(n_on, dtype=np.float64)
    mu_bkg = np.asanyarray(mu_bkg, dtype=np.float64)

    if method == "simple":
        func = _significance_simple
    elif method == "lima":
        func = _significance_lima
    elif method == "direct":
        func = _significance_direct
    else:
        raise ValueError("Invalid method: {}".format(method))

    # For low `n_on` values, don't try to compute a significance and return `NaN`.
    n_on = np.atleast_1d(n_on)
    mu_bkg = np.atleast_1d(mu_bkg)
    mask = n_on >= n_on_min
    s = np.ones_like(n_on) * np.nan
    s[mask] = func(n_on[mask], mu_bkg[mask])

    return s


def _significance_simple(n_on, mu_bkg):
    # TODO: check this formula against ???
    # Is this the limit from the on/off case for alpha -> 0?
    excess = n_on - mu_bkg
    bkg_err = np.sqrt(mu_bkg)
    return excess / bkg_err


def _significance_lima(n_on, mu_bkg):
    sign = np.sign(n_on - mu_bkg)
    val = np.sqrt(2) * np.sqrt(n_on * np.log(n_on / mu_bkg) - n_on + mu_bkg)
    return sign * val


def significance_on_off(n_on, n_off, alpha, method="lima"):
    r"""Compute significance of an on-off observation.

    The default ``method="lima"`` gives the significance estimate corresponding
    to equation (17) from the Li & Ma paper [1]_.

    For ``method="simple"``, the significance estimate is given by Equation (5)
    from the Li & Ma paper [1]_. It is somewhat biased, but has the advantage
    of being analytically invertible, which is useful for sensitivity computations.

    .. math::
        S_{simple} = \mu_{excess} / \Delta\mu_{excess}

        \mu_{excess} = n_{on} - \alpha n_{off}

        \Delta\mu_{excess} = \sqrt{n_{on} + \alpha ^ 2 n_{off}}

    Parameters
    ----------
    n_on : array_like
        Observed number of counts in the on region
    n_off : array_like
        Observed number of counts in the off region
    alpha : array_like
        On / off region exposure ratio for background events
    method : {"lima", "simple"}
        Method for significance estimation

    Returns
    -------
    significance : `~numpy.ndarray`
        Significance estimate

    References
    ----------
    .. [1] Li and Ma, "Analysis methods for results in gamma-ray astronomy",
       `Link <https://ui.adsabs.harvard.edu/abs/1983ApJ...272..317L>`_

    See Also
    --------
    significance, excess_matching_significance_on_off

    Examples
    --------
    >>> significance_on_off(n_on=10, n_off=20, alpha=0.1, method='lima')
    3.6850322319420274
    >>> significance_on_off(n_on=10, n_off=20, alpha=0.1, method='simple')
    2.5048971643405982
    """
    n_on = np.asanyarray(n_on, dtype=np.float64)
    n_off = np.asanyarray(n_off, dtype=np.float64)
    alpha = np.asanyarray(alpha, dtype=np.float64)

    with np.errstate(invalid="ignore", divide="ignore"):
        if method == "simple":
            return _significance_simple_on_off(n_on, n_off, alpha)
        elif method == "lima":
            return _significance_lima_on_off(n_on, n_off, alpha)
        elif method == "direct":
            return _significance_direct_on_off(n_on, n_off, alpha)
        else:
            raise ValueError("Invalid method: {}".format(method))


def _significance_simple_on_off(n_on, n_off, alpha):
    """Compute significance with the simple Equation (5) from Li & Ma."""
    excess_ = excess(n_on, n_off, alpha)
    excess_error_ = excess_error(n_on, n_off, alpha)

    return excess_ / excess_error_


def _significance_lima_on_off(n_on, n_off, alpha):
    r"""Compute significance with the Li & Ma formula (17)."""
    sign = np.sign(excess(n_on, n_off, alpha))

    tt = (alpha + 1) / (n_on + n_off)
    ll = n_on * np.log(n_on * tt / alpha)
    mm = n_off * np.log(n_off * tt)
    val = np.sqrt(np.abs(2 * (ll + mm)))

    return sign * val


def _significance_direct(n_on, mu_bkg):
    """Compute significance directly via Poisson probability.

    Reference: TODO (is this ever used?)
    """
    # Compute tail probability to see n_on or more counts
    # Note that we're using ``k = n_on - 1`` to get the probability
    # for n_on included or more, because `poisson.sf(k)` returns the
    # probability for more than k, with k excluded
    # For `n_on = 0` this returns `
    probability = poisson.sf(n_on - 1, mu_bkg)

    # Convert probability to a significance
    return norm.isf(probability)


def _significance_direct_on_off(n_on, n_off, alpha):
    """Compute significance directly via Poisson probability.

    Reference: https://ui.adsabs.harvard.edu/abs/1993NIMPA.328..570A

    You can use this method for small n_on < 10.
    In this case the Li & Ma formula isn't correct any more.
    """
    f = np.math.factorial

    # Not vectorised, only works for scalar, integer inputs
    n_on = int(n_on)
    n_off = int(n_off)

    # Compute tail probability to see n_on or more counts
    probability = 1
    for n in range(0, n_on):
        term_1 = alpha ** n / (1 + alpha) ** (n_off + n + 1)
        term_2 = f(n_off + n) / (f(n) * f(n_off))
        probability -= term_1 * term_2

    # Convert probability to a significance
    return norm.isf(probability)


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
        raise ValueError("Non-positive excess_error: {}".format(excess_error))

    if excess >= 0.0:
        zeta = excess / excess_error
        value = zeta / np.sqrt(2.0)
        integral = (1.0 + erf(value)) / 2.0
        integral2 = 1.0 - conf_level1 * integral
        value_old = value
        value_new = value_old + 0.01
        if integral > integral2:
            value_new = 0.0
        integral = (1.0 + erf(value_new)) / 2.0
    else:
        zeta = -excess / excess_error
        value = zeta / np.sqrt(2.0)
        integral = 1 - (1.0 + erf(value)) / 2.0
        integral2 = 1.0 - conf_level1 * integral
        value_old = value
        value_new = value_old + 0.01
        integral = (1.0 + erf(value_new)) / 2.0

    # The 1st Loop is for Speed & 2nd For Precision
    while integral < integral2:
        value_old = value_new
        value_new = value_new + 0.01
        integral = (1.0 + erf(value_new)) / 2.0
    value_new = value_old + 0.0000001
    integral = (1.0 + erf(value_new)) / 2.0

    while integral < integral2:
        value_new = value_new + 0.0000001
        integral = (1.0 + erf(value_new)) / 2.0
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
        raise ValueError("Invalid method: {}".format(method))


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
        raise ValueError("Invalid method: {}".format(method))


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
    n_on = fsolve(target_significance, n_on_guess, **opts)
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
    n_on = fsolve(target_significance, n_on_guess, **opts)
    return n_on - background(n_off, alpha)


_excess_matching_significance_lima = np.vectorize(_excess_matching_significance_lima)
_excess_matching_significance_on_off_lima = np.vectorize(
    _excess_matching_significance_on_off_lima
)
