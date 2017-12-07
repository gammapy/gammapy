# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Conversion functions for test statistic <-> significance <-> probability.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

# TODO: make all the other methods private?
# need to transfer the info from their docstrings to `convert_likelihood` first!
# TODO: check with MC study if there's a factor 2 error in the p-values
# because half of the TS values are exactly zero when fitting e.g. source extension.
# Do we need to introduce a bool "one_sided" or "hard_limit"?


__all__ = ['convert_likelihood',
           'significance_to_probability_normal',
           'probability_to_significance_normal',
           'probability_to_significance_normal_limit',
           'significance_to_probability_normal_limit',
           ]


def convert_likelihood(to, probability=None, significance=None,
                       ts=None, chi2=None, df=None):
    """Convert between various equivalent likelihood measures.

    TODO: don't use ``chi2`` with this function at the moment ...
    I forgot that one also needs the number of data points to
    compute ``ts``:
    http://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test#Calculating_the_test-statistic
    Probably it's best to split this out into a separate function
    or just document how users should compute ``ts`` before calling this
    function if they have ``chi2``.


    This function uses the ``sf`` and ``isf`` methods of the
    `~scipy.stats.norm` and `~scipy.stats.chi2` distributions
    to convert between various equivalent ways to quote a likelihood.

    - ``sf`` means "survival function", which is the "tail probability"
      of the distribution and is defined as ``1 - cdf``, where ``cdf``
      is the "cumulative distribution function".
    - ``isf`` is the inverse survival function.

    The relation between the quantities can be summarised as:

    - significance <-- normal distribution ---> probability
    - probability <--- chi2 distribution with df ---> ts
    - ts = chi2 / df

    So supporting both ``ts`` and ``chi2`` in this function is redundant,
    it's kept as a convenience for users that have a ``ts`` value from
    a Poisson likelihood fit and users that have a ``chi2`` value from
    a chi-square fit.

    Parameters
    ----------
    to : {'probability', 'ts', 'significance', 'chi2'}
        Which quantity you want to compute.
    probability, significance, ts, chi2 : array_like
        Input quantity value ... mutually exclusive, pass exactly one!
    df : array_like
        Difference in number of degrees of freedom between
        the alternative and the null hypothesis model.

    Returns
    -------
    value : `numpy.ndarray`
        Output value as requested by the input ``to`` parameter.

    Notes
    -----

    **TS computation**

    Under certain assumptions Wilk's theorem say that the likelihood ratio
    ``TS = 2 (L_alt - L_null)`` has a chi-square distribution with ``ndf``
    degrees of freedom in the null hypothesis case, where
    ``L_alt`` and ``L_null`` are the log-likelihoods in the null and alternative
    hypothesis and ``ndf`` is the difference in the number of freedom in those models.

    Note that the `~gammapy.stats.cash` statistic already contains the factor 2,
    i.e. you should compute ``TS`` as ``TS = cash_alt - cash_null``.

    - https://en.wikipedia.org/wiki/Chi-squared_distribution
    - http://docs.scipy.org/doc/scipy-dev/reference/generated/scipy.stats.chi2.html
    - https://en.wikipedia.org/wiki/Likelihood-ratio_test
    - https://adsabs.harvard.edu/abs/1979ApJ...228..939C
    - https://adsabs.harvard.edu/abs/2009A%26A...495..989S

    **Physical limits**

    ``probability`` is the one-sided `p-value`, e.g. `significance=3`
    corresponds to `probability=0.00135`.

    TODO: check if this gives correct coverage for cases with hard physical limits,
    e.g. when fitting TS of extended sources vs. point source and in half of the
    cases ``TS=0`` ... I suspect coverage might not be OK and we need to add an
    option to this function to handle those cases!

    Examples
    --------
    Here's some examples how to compute the ``probability`` or ``significance``
    for a given observed ``ts`` or ``chi2``:

    >>> from gammapy.stats import convert_likelihood
    >>> convert_likelihood(to='probability', ts=10, df=2)
    0.0067379469990854679
    >>> convert_likelihood(to='significance', chi2=19, df=7)
    2.4004554920435521

    Here's how to do the reverse, compute the ``ts`` or ``chi2`` that would
    result in a given ``probability`` or ``significance``.

    >>> convert_likelihood(to='ts', probability=0.01, df=1)
    6.6348966010212171
    >>> convert_likelihood(to='chi2', significance=3, df=10)
    28.78498865156606
    """
    from scipy.stats import norm as norm_distribution
    from scipy.stats import chi2 as chi2_distribution

    # ---> Check inputs are OK!
    # ---> This is a function that will be used interactively by end-users,
    # ---> so we want good error messages if they use it correctly.

    # Check that the output `to` parameter is valid
    valid_quantities = ['probability', 'ts', 'significance', 'chi2']
    if to not in valid_quantities:
        msg = 'Invalid parameter `to`: {}\n'.format(to)
        msg += 'Valid options are: {}'.format(valid_quantities)
        raise ValueError(msg)

    # Check that the input is valid
    _locals = locals().copy()
    input_values = [_ for _ in valid_quantities
                    if _locals[_] is not None]
    if len(input_values) != 1:
        msg = 'You have to pass exactly one of the valid input quantities: '
        msg += ', '.join(valid_quantities)
        msg += '\nYou passed: '
        if len(input_values) == 0:
            msg += 'none'
        else:
            msg += ', '.join(input_values)
        raise ValueError(msg)

    input_type = input_values[0]
    input_value = locals()[input_type]

    # Check that `df` is given if it's required for the computation
    if any(_ in ['ts', 'chi2'] for _ in [input_type, to]) and df is None:
        msg = 'You have to specify the number of degrees of freedom '
        msg += 'via the `df` parameter.'
        raise ValueError(msg)


    # ---> Compute the requested quantity
    # ---> By now we know the inputs are OK.

    # Compute equivalent `ts` for `chi2` ... after this
    # the code will only handle the `ts` input case,
    # i.e. conversions: significance <-> probability <-> ts
    if chi2 is not None:
        ts = chi2 / df

    # A note that might help you understand the nested if-else-statement:
    # The quantities `probability`, `significance`, `ts` and `chi2`
    # form a graph with `probability` at the center.
    # There might be functions directly relating the other quantities
    # in general or in certain limits, but the computation here
    # always proceeds via `probability` as a one- or two-step process.

    if to == 'significance':
        if ts is not None:
            probability = chi2_distribution.sf(ts, df)
        return norm_distribution.isf(probability)

    elif to == 'probability':
        if significance is not None:
            return norm_distribution.sf(significance)
        else:
            return chi2_distribution.sf(ts, df)

    elif to == 'ts':
        # Compute a probability if needed
        if significance is not None:
            probability = norm_distribution.sf(significance)

        return chi2_distribution.isf(probability, df)

    elif to == 'chi2':
        if ts is not None:
            return df * ts
        # Compute a probability if needed
        if significance is not None:
            probability = norm_distribution.sf(significance)

        return chi2_distribution.isf(probability, df)


def significance_to_probability_normal(significance):
    """Convert significance to one-sided tail probability.

    Parameters
    ----------
    significance : array_like
        Significance

    Returns
    -------
    probability : ndarray
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
    from scipy.stats import norm
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
    from scipy.stats import norm
    return norm.isf(probability)


def _p_to_s_direct(probability, one_sided=True):
    """Direct implementation of p_to_s for checking.

    Reference: RooStats User Guide Equations (6,7).
    """
    from scipy.special import erfinv
    probability = 1 - probability  # We want p to be the tail probability
    temp = np.where(one_sided, 2 * probability - 1, probability)
    return np.sqrt(2) * erfinv(temp)


def _s_to_p_direct(significance, one_sided=True):
    """Direct implementation of s_to_p for checking.

    Note: _p_to_s_direct was solved for p.
    """
    from scipy.special import erf
    temp = erf(significance / np.sqrt(2))
    probability = np.where(one_sided, (temp + 1) / 2., temp)
    return 1 - probability  # We want p to be the tail probability


def probability_to_significance_normal_limit(probability):
    """Convert tail probability to significance
    in the limit of small p and large s.

    Reference: Equation (4) of
    http://adsabs.harvard.edu/abs/2007physics...2156C
    They say it is better than 1% for s > 1.6.

    Asymptotically: s ~ sqrt(-log(p))
    """
    u = -2 * np.log(probability * np.sqrt(2 * np.pi))
    return np.sqrt(u - np.log(u))


def significance_to_probability_normal_limit(significance, guess=1e-100):
    """Convert significance to tail probability
    in the limit of small p and large s.

    See p_to_s_limit docstring
    Note: s^2 = u - log(u) can't be solved analytically.
    """
    from scipy.optimize import fsolve

    def f(probability):
        if probability > 0:
            return probability_to_significance_normal_limit(probability) - significance
        else:
            return 1e100

    return fsolve(f, guess)
