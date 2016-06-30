# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Common fit statistics used in gamma-ray astronomy.

References
----------

Results were tested against results from the
`Sherpa <http://cxc.harvard.edu/sherpa/>`_ and
`XSpec <https://heasarc.gsfc.nasa.gov/xanadu/xspec/>`_
X-ray analysis packages.

Each function contains references for the implemented formulae,
to get an overview have a look at the
`Sherpa statistics page <http://cxc.cfa.harvard.edu/sherpa/statistics>`_ or the
`XSpec manual statistics page <http://heasarc.nasa.gov/xanadu/xspec/manual/XSappendixStatistics.html>`_.

Examples
--------

All functions compute per-bin statistics.
If you want the summed statistics for all bins,
call sum on the output array yourself.
Here's an example for the `~cash` statistic::

    >>> from gammapy.stats import cash
    >>> data = [3, 5, 9]
    >>> model = [3.3, 6.8, 9.2]
    >>> cash(data, model)
    array([ -0.56353481,  -5.56922612, -21.54566271])
    >>> cash(data, model).sum()
    -27.678423645645118
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

__all__ = ['cash', 'cstat', 'wstat', 'lstat', 'pgstat',
           'chi2', 'chi2constvar', 'chi2datavar',
           'chi2gehrels', 'chi2modvar', 'chi2xspecvar',
           ]

N_OBSERVED_MIN = 1e-25


def cash(n_observed, mu_predicted):
    r"""Cash statistic, for Poisson data.

    The Cash statistic is defined as:

    .. math::
        C = 2 \left( n_{observed} - n_{observed} \log \mu_{prediced} \right)

    and :math:`C = 0` where :math:`\mu <= 0`.

    Parameters
    ----------
    n_observed : array_like
        Observed counts
    mu_predicted : array_like
        Expected counts

    Returns
    -------
    stat : ndarray
        Statistic per bin

    References
    ----------
    * `Sherpa statistics page section on the Cash statistic
      <http://cxc.cfa.harvard.edu/sherpa/statistics/#cash>`_
    * `Sherpa help page on the Cash statistic
      <http://cxc.harvard.edu/sherpa/ahelp/cash.html>`_
    * `Cash 1979, ApJ 228, 939
      <http://adsabs.harvard.edu/abs/1979ApJ...228..939C>`_
    """
    n_observed = np.asanyarray(n_observed, dtype=np.float64)
    mu_predicted = np.asanyarray(mu_predicted, dtype=np.float64)

    stat = 2 * (mu_predicted - n_observed * np.log(mu_predicted))
    stat = np.where(mu_predicted > 0, stat, 0)
    return stat


def cstat(n_observed, mu_predicted, n_observed_min=N_OBSERVED_MIN):
    r"""C statistic, for Poisson data.

    The C statistic is defined as

    .. math::
        C = 2 \left[ \mu_{prediced} - n_{observed} + n_{observed}
            (\log(n_{observed}) - log(\mu_{prediced}) \right]

    and :math:`C = 0` where :math:`\mu_{observed} <= 0`.

    ``n_observed_min`` handles the case where ``n_observed`` is 0 or less and
    the log cannot be taken.

    Parameters
    ----------
    n_observed : array_like
        Observed counts
    mu_predicted : array_like
        Expected counts
    n_observed_min : array_like
        ``n_observed`` = ``n_observed_min`` where ``n_observed`` <= ``n_observed_min.``

    Returns
    -------
    stat : ndarray
        Statistic per bin

    References
    ----------
    * `Sherpa stats page section on the C statistic
      <http://cxc.cfa.harvard.edu/sherpa/statistics/#cstat>`_
    * `Sherpa help page on the C statistic
      <http://cxc.harvard.edu/sherpa/ahelp/cash.html>`_
    * `Cash 1979, ApJ 228, 939
      <http://adsabs.harvard.edu/abs/1979ApJ...228..939C>`_
    """
    n_observed = np.asanyarray(n_observed, dtype=np.float64)
    mu_predicted = np.asanyarray(mu_predicted, dtype=np.float64)
    n_observed_min = np.asanyarray(n_observed_min, dtype=np.float64)

    n_observed = np.where(n_observed <= n_observed_min, n_observed_min, n_observed)

    term1 = np.log(n_observed) - np.log(mu_predicted)
    stat = 2 * (mu_predicted - n_observed + n_observed * term1)
    stat = np.where(mu_predicted > 0, stat, 0)

    return stat


def wstat(n_on, n_bkg, mu_signal):
    r"""W statistic, for Poisson data with Poisson background.

    Consult the reference page for a definition of WStat.

    Parameters
    ----------
    n_on : array_like
        Total observed counts
    n_bkg : array_like
        Background counts
    mu_signal : array_like
        Signal expected counts

    Returns
    -------
    stat : ndarray
        Statistic per bin

    References
    ----------
    * `XSPEC page on Poisson data with Poisson background
    <http://heasarc.nasa.gov/xanadu/xspec/manual/XSappendixStatistics.html>`_
    """
    # Mute numpy errors since they are expected and treated in the end
    original_state = np.geterr()
    np.seterr(all='ignore')

    n_on = np.asanyarray(n_on, dtype=np.float64)
    n_bkg = np.asanyarray(n_bkg, dtype=np.float64)
    mu_signal = np.asanyarray(mu_signal, dtype=np.float64)

    # variable names are geared to the names on the XSPEC reference page
    d_term1 = 2 * mu_signal - n_on - n_bkg
    d_term2 = 8 * n_bkg * mu_signal
    d = np.sqrt(d_term1**2 + d_term2)
    
    f_temp = n_on + n_bkg - 2 * mu_signal
    f_temp_plus = f_temp + d
    f_temp_minus = f_temp - d

    f_num = np.where(f_temp_plus > 0, f_temp_plus, f_temp_minus)
    f_den = 4
    mu_background = f_num / f_den

    term1 = mu_signal + 2 * mu_background
    term2 = n_on * np.log(mu_signal + mu_background)
    term3 = n_bkg * np.log(mu_background)
    term4 = n_on * (1-np.log(n_on)) + n_bkg * (1-np.log(n_bkg))

    stat = 2 * (term1 - term2 - term3 - term4)
    # This may contain nan values where n_on or n_bkg are zero 

    np.seterr(**original_state)
    idx = np.isnan(stat)
    if idx.any():
        stat[idx] = 0
        special_cases = np.zeros(len(stat))
        for pos in np.where(idx)[0]:
            if n_on[pos] == 0:
                statval = mu_signal[pos] - n_bkg[pos] * np.log(0.5)
            elif n_bkg[pos] == 0:
                if mu_signal[pos] < (n_on[pos] / 2):
                    statval = - mu_signal[pos] - n_on[pos] * np.log(0.5)
                else:
                    temp = (np.log(n_on[pos]) - np.log(mu_signal[pos]) - 1)
                    statval = mu_signal[pos] + n_on[pos] * temp
            else:
                raise ValueError("This should never be reached")
            special_cases[pos] = statval

        stat = stat + special_cases

    return stat


def lstat():
    r"""L statistic, for Poisson data with Poisson background (Bayesian).

    Reference: http://heasarc.nasa.gov/xanadu/xspec/manual/XSappendixStatistics.html
    """
    pass


def pgstat():
    r"""PG statistic, for Poisson data with Gaussian background.

    Reference: http://heasarc.nasa.gov/xanadu/xspec/manual/XSappendixStatistics.html
    """
    pass


def chi2(N_S, B, S, sigma2):
    r"""Chi-square statistic with user-specified variance.

     .. math::

         \chi^2 = \frac{(N_S - B - S) ^ 2}{\sigma ^ 2}

    Parameters
    ----------
    N_S : array_like
        Number of observed counts
    B : array_like
        Model background
    S : array_like
        Model signal
    sigma2 : array_like
        Variance

    Returns
    -------
    stat : ndarray
        Statistic per bin

    References
    ----------
    * Sherpa stats page (http://cxc.cfa.harvard.edu/sherpa/statistics/#chisq)
    """
    N_S = np.asanyarray(N_S, dtype=np.float64)
    B = np.asanyarray(B, dtype=np.float64)
    S = np.asanyarray(S, dtype=np.float64)
    sigma2 = np.asanyarray(sigma2, dtype=np.float64)

    stat = (N_S - B - S) ** 2 / sigma2

    return stat


def chi2constvar(N_S, N_B, A_S, A_B):
    r"""Chi-square statistic with constant variance.
    """
    N_S = np.asanyarray(N_S, dtype=np.float64)
    N_B = np.asanyarray(N_B, dtype=np.float64)
    A_S = np.asanyarray(A_S, dtype=np.float64)
    A_B = np.asanyarray(A_B, dtype=np.float64)

    alpha2 = (A_S / A_B) ** 2
    # Need to mulitply with np.ones_like(N_S) here?
    sigma2 = (N_S + alpha2 * N_B).mean()

    stat = chi2(N_S, A_B, A_S, sigma2)

    return stat


def chi2datavar(N_S, N_B, A_S, A_B):
    r"""Chi-square statistic with data variance.
    """
    N_S = np.asanyarray(N_S, dtype=np.float64)
    N_B = np.asanyarray(N_B, dtype=np.float64)
    A_S = np.asanyarray(A_S, dtype=np.float64)
    A_B = np.asanyarray(A_B, dtype=np.float64)

    alpha2 = (A_S / A_B) ** 2
    sigma2 = N_S + alpha2 * N_B

    stat = chi2(N_S, A_B, A_S, sigma2)

    return stat


def chi2gehrels(N_S, N_B, A_S, A_B):
    r"""Chi-square statistic with Gehrel's variance.
    """
    N_S = np.asanyarray(N_S, dtype=np.float64)
    N_B = np.asanyarray(N_B, dtype=np.float64)
    A_S = np.asanyarray(A_S, dtype=np.float64)
    A_B = np.asanyarray(A_B, dtype=np.float64)

    alpha2 = (A_S / A_B) ** 2
    sigma_S = 1 + np.sqrt(N_S + 0.75)
    sigma_B = 1 + np.sqrt(N_B + 0.75)
    sigma2 = sigma_S ** 2 + alpha2 * sigma_B ** 2

    stat = chi2(N_S, A_B, A_S, sigma2)

    return stat


def chi2modvar(S, B, A_S, A_B):
    r"""Chi-square statistic with model variance.
    """
    S = np.asanyarray(S, dtype=np.float64)
    B = np.asanyarray(B, dtype=np.float64)
    A_S = np.asanyarray(A_S, dtype=np.float64)
    A_B = np.asanyarray(A_B, dtype=np.float64)

    stat = chi2datavar(S, B, A_S, A_B)

    return stat


def chi2xspecvar(N_S, N_B, A_S, A_B):
    r"""Chi-square statistic with XSPEC variance.
    """
    N_S = np.asanyarray(N_S, dtype=np.float64)
    N_B = np.asanyarray(N_B, dtype=np.float64)
    A_S = np.asanyarray(A_S, dtype=np.float64)
    A_B = np.asanyarray(A_B, dtype=np.float64)

    # TODO: is this correct?
    mask = (N_S < 1) | (N_B < 1)
    # _stat = np.empty_like(mask, dtype='float')
    # _stat[mask] = 1
    stat = np.where(mask, 1, chi2datavar(N_S, N_B, A_S, A_B))

    return stat
