# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Poisson statistics computations for these two cases.

* a measured number of counts ``n_on`` and known background
* a measured number of counts ``n_on`` in an on region
  and a second count measurement ``n_off`` in an excess-free region.

TODO: More detailed description here.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy import sign, log, sqrt

__all__ = ['background', 'background_error',
           'excess', 'excess_error',
           'significance', 'significance_on_off',
           'sensitivity', 'sensitivity_on_off',
           ]

__doctest_skip__ = ['*']


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
    background : ndarray
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
    r"""Estimate standard error on background
    in the on region from an off-region observation.

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
    background : ndarray
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

    return alpha * sqrt(n_off)


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
    excess : ndarray
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
    r"""Estimate standard error on excess
    in the on region for an on-off observation.

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
    excess_error : ndarray
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
    return sqrt(variance)


# TODO: rename this function to something more explicit.
# It currently has the same name as the `gammapy/stats/significance.py`
# and shadows in in `gammapy/stats/__init.py`
# Maybe `significance_poisson`?
def significance(n_on, mu_bkg, method='lima', n_on_min=1):
    r"""Compute significance for an observed number of counts and known background.

    The simple significance estimate :math:`S_{simple}` is given by

    .. math ::

        S_{simple} = (n_{on} - \mu_{bkg}) / \sqrt{\mu_{bkg}}

    The Li & Ma significance estimate corresponds to the Li & Ma formula (17)
    in the limiting case of known background :math:`\mu_{bkg} = \alpha \times n_{off}`
    with :math:`\alpha \to 0`.
    The following formula for :math:`S_{lima}` was obtained with Mathematica:

    .. math ::

        S_{lima} = \left[ 2 n_{on} \log \left( \frac{n_{on}}{\mu_{bkg}} \right) - n_{on} + \mu_{bkg} \right] ^ {1/2}


    Parameters
    ----------
    n_on : array_like
        Observed number of counts
    mu_bkg : array_like
        Known background level
    method : str
        Select method: 'lima' or 'simple'
    n_on_min : float
        Minimum ``n_on`` (return ``NaN`` for smaller values)

    Returns
    -------
    significance : ndarray
        Significance according to the method chosen.

    References
    ----------
    .. [1] Li and Ma, "Analysis methods for results in gamma-ray astronomy",
       `Link <http://adsabs.harvard.edu/abs/1983ApJ...272..317L>`_

    See Also
    --------
    excess, significance_on_off

    Examples
    --------
    >>> significance(n_on=11, mu_bkg=9, method='lima')
    0.64401498442763649
    >>> significance(n_on=11, mu_bkg=9, method='simple')
    0.66666666666666663
    >>> significance(n_on=7, mu_bkg=9, method='lima')
    -0.69397262486881672
    >>> significance(n_on=7, mu_bkg=9, method='simple')
    -0.66666666666666663
    """
    n_on = np.asanyarray(n_on, dtype=np.float64)
    mu_bkg = np.asanyarray(mu_bkg, dtype=np.float64)

    if method == 'simple':
        func = _significance_simple
    elif method == 'lima':
        func = _significance_lima
    elif method == 'direct':
        func = _significance_direct
    else:
        raise ValueError('Invalid method: {}'.format(method))

    # For low `n_on` values, don't try to compute a significance and return `NaN`.
    n_on = np.atleast_1d(n_on)
    mu_bkg = np.atleast_1d(mu_bkg)
    mask = (n_on >= n_on_min)
    s = np.ones_like(n_on) * np.nan
    s[mask] = func(n_on[mask], mu_bkg[mask])

    return s


def _significance_simple(n_on, mu_bkg):
    # TODO: check this formula against ???
    excess = n_on - mu_bkg
    bkg_err = sqrt(mu_bkg)
    return excess / bkg_err


def _significance_lima(n_on, mu_bkg):
    # import IPython; IPython.embed()

    term_a = sign(n_on - mu_bkg) * sqrt(2)
    term_b = sqrt(n_on * log(n_on / mu_bkg) - n_on + mu_bkg)
    return term_a * term_b


def _significance_direct(n_on, mu_bkg):
    """Compute significance directly via Poisson probability.

    Use this method for small ``n_on < 10``.
    In this case the Li & Ma formula isn't correct any more.

    TODO: add large unit test coverage (where is it numerically precise enough)?
    TODO: check coverage with MC simulation

    I'm getting a positive significance for zero observed counts and small mu_bkg.
    That doesn't make too much sense ...

    >>> stats.poisson._significance_direct(0, 2)
    -1.1015196284987503

    >>> stats.poisson._significance_direct(0, 0.1)
    1.309617799458493

    """
    from scipy.stats import norm, poisson

    # Compute tail probability to see n_on or more counts
    probability = poisson.sf(n_on, mu_bkg)

    # Convert probability to a significance
    significance = norm.isf(probability)

    return significance


def significance_on_off(n_on, n_off, alpha, method='lima',
                        neglect_background_uncertainty=False):
    r"""Compute significance of an on-off observation.

    TODO: describe available methods.

    Parameters
    ----------
    n_on : array_like
        Observed number of counts in the on region
    n_off : array_like
        Observed number of counts in the off region
    alpha : array_like
        On / off region exposure ratio for background events
    method : {'lima', 'simple', 'direct'}
        Select method

    Returns
    -------
    significance : array
        Significance according to the method chosen.

    References
    ----------
    .. [1] Li and Ma, "Analysis methods for results in gamma-ray astronomy",
       `Link <http://adsabs.harvard.edu/abs/1983ApJ...272..317L>`_

    See Also
    --------
    significance, sensitivity_on_off

    Examples
    --------
    >>> significance_on_off(n_on=10, n_off=20, alpha=0.1, method='lima')
    3.6850322319420274
    >>> significance_on_off(n_on=10, n_off=20, alpha=0.1, method='simple')
    2.5048971643405982
    >>> significance_on_off(n_on=10, n_off=20, alpha=0.1, method='direct')
    3.5281644971409953
    """
    n_on = np.asanyarray(n_on, dtype=np.float64)
    n_off = np.asanyarray(n_off, dtype=np.float64)
    alpha = np.asanyarray(alpha, dtype=np.float64)

    if method == 'simple':
        if neglect_background_uncertainty:
            mu_bkg = background(n_off, alpha)
            return _significance_simple(n_on, mu_bkg)
        else:
            return _significance_simple_on_off(n_on, n_off, alpha)
    elif method == 'lima':
        if neglect_background_uncertainty:
            mu_bkg = background(n_off, alpha)
            return _significance_lima(n_on, mu_bkg)
        else:
            return _significance_lima_on_off(n_on, n_off, alpha)
    elif method == 'direct':
        if neglect_background_uncertainty:
            mu_bkg = background(n_off, alpha)
            return _significance_direct(n_on, mu_bkg)
        else:
            return _significance_direct_on_off(n_on, n_off, alpha)
    else:
        raise ValueError('Invalid method: {}'.format(method))


def _significance_simple_on_off(n_on, n_off, alpha):
    r"""Compute significance with a simple, somewhat biased formula.

    .. math::

        S = \mu_{excess} / \Delta\mu_{excess}

        where

        \mu_{excess} = n_{on} - \alpha \times n_{off}

        \Delta\mu_{excess} = \sqrt{n_{on} + \alpha ^ 2 \times n_{off}}

    Notes
    -----
    This function implements formula (5) of Li & Ma.
    Li & Ma show that it is somewhat biased,
    but it does have the advantage of being analytically invertible,
    i.e. there is an analytical formula for sensitivity,
    which is often used in practice.
    """
    excess_ = excess(n_on, n_off, alpha)
    excess_error_ = excess_error(n_on, n_off, alpha)

    return excess_ / excess_error_


def _significance_lima_on_off(n_on, n_off, alpha):
    r"""Compute significance with the Li & Ma formula (17)."""
    temp = (alpha + 1) / (n_on + n_off)
    l = n_on * log(n_on * temp / alpha)
    m = n_off * log(n_off * temp)
    e = excess(n_on, n_off, alpha)
    sign = np.where(e > 0, 1, -1)

    return sign * sqrt(abs(2 * (l + m)))


def _significance_direct_on_off(n_on, n_off, alpha):
    """Compute significance directly via Poisson probability.

    Use this method for small n_on < 10.
    In this case the Li & Ma formula isn't correct any more.

    * TODO: add reference
    * TODO: add large unit test coverage (where is it numerically precise enough)?
    * TODO: check coverage with MC simulation
    * TODO: implement in Cython and vectorize n_on (accept numpy  array n_on as input)
    """
    from math import factorial as fac
    from scipy.stats import norm

    # Compute tail probability to see n_on or more counts
    probability = 1
    for n in range(0, n_on):
        term_1 = alpha ** n / (1 + alpha) ** (n_off + n + 1)
        term_2 = fac(n_off + n) / (fac(n) * fac(n_off))
        probability -= term_1 * term_2

    # Convert probability to a significance
    significance = norm.isf(probability)

    return significance


def sensitivity(mu_bkg, significance, quantity='excess', method='lima'):
    r"""Compute sensitivity.

    TODO: document what "sensitivity" is ... excess or n_on.
    Use `quantity` parameter or decide on one quanity and remove that option.

    Parameters
    ----------
    mu_bkg : array_like
        Known background level
    quantity : {'excess', 'n_on'}
        Select output quantity
    method : {'lima', 'simple'}
        Select method

    Returns
    -------
    sensitivity : ndarray
        Sensitivity according to the method chosen.

    See Also
    --------
    sensitivity_on_off

    Examples
    --------
    >>> # sensitivity(mu_bkg=0.2, significance=5, method='lima')
    TODO
    >>> # sensitivity(mu_bkg=0.2, significance=5, method='simple')
    TODO
    """
    mu_bkg = np.asanyarray(mu_bkg, dtype=np.float64)
    significance = np.asanyarray(significance, dtype=np.float64)

    if method == 'simple':
        return _sensitivity_simple(mu_bkg, significance)
    elif method == 'lima':
        return _sensitivity_lima(mu_bkg, significance)
    else:
        raise ValueError('Invalid method: {}'.format(method))


def _sensitivity_simple(mu_bkg, significance):
    raise NotImplementedError


def _sensitivity_lima(mu_bkg, significance):
    raise NotImplementedError


def sensitivity_on_off(n_off, alpha, significance, quantity='excess', method='lima'):
    r"""Compute sensitivity of an on-off observation.

    Parameters
    ----------
    n_off : array_like
        Observed number of counts in the off region
    alpha : array_like
        On / off region exposure ratio for background events
    significance : array_like
        Desired significance level
    quantity : {'excess', 'n_on'}
        Which output sensitivity quantity?
    method : {'lima', 'simple'}
        Which method?

    Returns
    -------
    sensitivity : `numpy.ndarray`
        Sensitivity according to the method chosen.

    See Also
    --------
    sensitivity, significance_on_off

    Examples
    --------
    >>> # sensitivity_on_off(n_off=20, alpha=0.1, significance=5, method='lima')
    TODO
    >>> # sensitivity_on_off(n_off=20, alpha=0.1, significance=5, method='simple')
    2.5048971
    """
    n_off = np.asanyarray(n_off, dtype=np.float64)
    alpha = np.asanyarray(alpha, dtype=np.float64)
    significance = np.asanyarray(significance, dtype=np.float64)

    if method == 'lima':
        n_on_sensitivity = _sensitivity_lima(n_off, alpha, significance)
    elif method == 'simple':
        n_on_sensitivity = _sensitivity_simple(n_off, alpha, significance)
    else:
        raise ValueError('Invalid method: {}'.format(method))

    if quantity == 'n_on':
        return n_on_sensitivity
    elif quantity == 'excess':
        return n_on_sensitivity - background(n_off, alpha)
    else:
        raise ValueError('Invalid quantity: {}'.format(quantity))


def _sensitivity_simple_on_off(n_off, alpha, significance):
    """Implements an analytical formula that can be easily obtained
    by solving the simple significance formula for n_on.
    """
    significance2 = significance ** 2
    determinant = significance2 + 4 * n_off * alpha * (1 + alpha)
    temp = significance2 + 2 * n_off * alpha
    n_on = 0.5 * (temp + significance * sqrt(abs(determinant)))
    return n_on


def _sensitivity_lima_on_off(n_off, alpha, significance):
    """Implements an iterative root finding method to solve the
    significance formula for n_on.

    TODO: in weird cases (e.g. on=0.1, off=0.1, alpha=0.001)
    fsolve does not find a solution.
    values < guess are often not found using this cost function.
    Find a way to make this function more robust and add plenty of tests.
    Maybe a better starting point estimate can help?
    """
    from scipy.optimize import fsolve

    def f(n_on, args):
        n_off, alpha, significance = args
        if n_on >= 0:
            return _significance_lima_on_off(n_on, n_off, alpha) - significance
        else:
            return 1e100

    # We need to loop over the array manually and call `fsolve` for
    # each item separately.
    n_on = np.empty_like(n_off)
    guess = _sensitivity_simple_on_off(n_off, alpha, significance) + background(n_off, alpha)
    data = enumerate(zip(guess.flat, n_off.flat, alpha.flat, significance.flat))
    for ii, guess_, n_off_, alpha_, significance_ in data:
        # guess = 1e-3
        n_on.flat[ii] = fsolve(f, guess_, args=(n_off_, alpha_, significance_))

    return n_on
