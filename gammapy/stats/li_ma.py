# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Li and Ma algorithm to compute parameter confidence limits and significance."""
import logging
import numpy as np

__all__ = [
    "lm_loglikelihood",
    "lm_dexcess_down",
    "lm_dexcess_up",
    "lm_dexcess",
    "lm_significance_on_off",
    "lm_significance"
]

log = logging.getLogger(__name__)


def lm_loglikelihood(n_on, n_off, alpha, signal):
    r"""Compute Li&Ma Log-likelihood.

    Note: This is a little trick to avoid problems for negative signal
    (note that this method is often called with signal=excess).
    In that case the Poisson signal + b0 can become negative
    and log(signal + b0) isn't defined, i.e. the likelihood cannot be computed.
    To avoid this problem, the on and off regions are switched

    For more information see :ref:`documentation <li_ma>`.

    Parameters
    ----------
    n_on : float
        Observed number of counts in the on region
    n_off : float
        Observed number of counts in the off region
    alpha : float
        On / off region exposure ratio for background events
    signal : float
        Number of excess in the on region

    Returns
    -------
    log_likelihood : `~numpy.ndarray`

    """
    if signal > 0:
        n1 = n_on
        n2 = n_off
        aa = alpha
    else:
        n2 = n_on
        n1 = n_off
        aa = 1.0 / alpha
        signal = -aa * signal

    # Maximum likelihood background estimate b0
    tt = (1+aa) * signal - aa * (n1+n2)
    delta = tt*tt + 4 * aa * (1+aa) * signal * n2
    b0 = (np.sqrt(delta) - tt) / (2 + 2*aa)
    # b0 should always be >= 0.
    # Just to avoid possible problems with rounding errors:
    if b0 < 0:
        b0 = 0.

    # Compute Poisson log likelihood r for two measurements: on / off
    ll = n1 * np.log(signal + b0) + n2 * np.log(b0 / aa) - (signal + b0) - b0 / aa
    return ll


def lm_dexcess_down(n_on, n_off, alpha, n_sig=1.):
    r"""Compute downward excess uncertainties of an on-off observation based on Li&Ma [1]. [LiMa1983]_

    Searches the average value of the signal for which the log-likelihood ratio is n_sig away

    Note: The required step in log likelihood for "nsig sigma errors" is offset = 0.5*n_sig^2
            http://wwwasdoc.web.cern.ch/wwwasdoc/minuit/node30.html

    For more information see :ref:`documentation <li_ma>`.

    Parameters
    ----------
    n_on : array_like
        Observed number of counts in the on region
    n_off : array_like
        Observed number of counts in the off region
    alpha : array_like
        On / off region exposure ratio for background events
    n_sig : float
        Number of sigma

    Returns
    -------
    significance : `~numpy.ndarray`
        Error estimate

     References
    ----------
    * Li and Ma, "Analysis methods for results in gamma-ray astronomy"
      <https://ui.adsabs.harvard.edu/abs/1983ApJ...272..317L>`_
    """
    scalar = False
    if np.isscalar(n_on):
        n_on = np.array([n_on])
        scalar = True
    if np.isscalar(n_off):
        n_off = np.array([n_off])
        scalar = True
    if np.isscalar(alpha):
        alpha = np.array([alpha])
        scalar = True
    n_on = np.asanyarray(n_on, dtype=np.float64)
    n_off = np.asanyarray(n_off, dtype=np.float64)
    alpha = np.asanyarray(alpha, dtype=np.float64)
    n_sig = np.full(n_on.shape, n_sig)

    # Optimal value of signal
    excess0 = n_on - n_off * alpha
    l0 = np.zeros(n_on.shape)
    for ii in range(len(n_on)):
        l0[ii] = lm_loglikelihood(n_on[ii], n_off[ii], alpha[ii], excess0[ii])

    # Search for signal such as -2 * log H > 1 * sig
    # i.e. find a signal > dExcess_Up so that we can
    # use an interval bisection algorithm to find dExcess_Up
    offset = np.full(n_on.shape, n_sig*n_sig*0.5)
    signal = excess0.copy() - 1
    ll = np.zeros(n_on.shape)
    for ii in range(len(n_on)):
        ll[ii] = lm_loglikelihood(n_on[ii], n_off[ii], alpha[ii], signal[ii])
        count = 0
        while (l0[ii]-ll[ii]) < offset[ii] and np.isfinite(ll[ii]) and count < 100:
            signal[ii] = (signal[ii]-excess0[ii]) * 2. + excess0[ii]
            ll[ii] = lm_loglikelihood(n_on[ii], n_off[ii], alpha[ii], signal[ii])
            count += 1

    # Dichotomic search (i.e. interval bisection algorithm)
    up = excess0.copy()
    low = signal.copy()
    signal = (up+low)/2.
    for ii in range(len(n_on)):
        ll[ii] = lm_loglikelihood(n_on[ii], n_off[ii], alpha[ii], signal[ii])
        count = 0.
        while np.abs(l0[ii]-ll[ii]-offset[ii]) > 1e-4 and np.isfinite(ll[ii]) and count < 100:
            if l0[ii]-ll[ii] > offset[ii]:
                low[ii] = signal[ii]
            else:
                up[ii] = signal[ii]
            signal[ii] = (up[ii]+low[ii])/2.
            ll[ii] = lm_loglikelihood(n_on[ii], n_off[ii], alpha[ii], signal[ii])
            count += 1

    if scalar:
        return np.abs(signal[0] - excess0[0])
    return np.abs(signal - excess0)


def lm_dexcess_up(n_on, n_off, alpha, n_sig=1.):
    r"""Compute upward excess uncertainties of an on-off observation based on Li&Ma [1].

    Searches the average value of the signal for which the log-likelihood ratio is n_sig away

    Note: The required step in log likelihood for "nsig sigma errors" is offset = 0.5*n_sig^2
            http://wwwasdoc.web.cern.ch/wwwasdoc/minuit/node30.html

    Parameters
    ----------
    n_on : array_like
        Observed number of counts in the on region
    n_off : array_like
        Observed number of counts in the off region
    alpha : array_like
        On / off region exposure ratio for background events
    n_sig : float
        Number of sigma

    Returns
    -------
    significance : `~numpy.ndarray`
        Error estimate

    References
    ----------
    .. [1] Li and Ma, "Analysis methods for results in gamma-ray astronomy",
       `Link <https://ui.adsabs.harvard.edu/abs/1983ApJ...272..317L>`_
    """
    scalar = False
    if np.isscalar(n_on):
        n_on = np.array([n_on])
        scalar = True
    if np.isscalar(n_off):
        n_off = np.array([n_off])
        scalar = True
    if np.isscalar(alpha):
        alpha = np.array([alpha])
        scalar = True
    n_on = np.asarray(n_on, dtype=np.float64)
    n_off = np.asanyarray(n_off, dtype=np.float64)
    alpha = np.asanyarray(alpha, dtype=np.float64)
    n_sig = np.full(n_on.shape, n_sig)

    # Optimal value of signal
    excess0 = n_on - n_off * alpha
    l0 = np.zeros(n_on.shape)
    for ii in range(len(n_on)):
        l0[ii] = lm_loglikelihood(n_on[ii], n_off[ii], alpha[ii], excess0[ii])

    # Search for signal such as -2 * log H > 1 * sig
    # i.e. find a signal > dExcess_Up so that we can
    # use an interval bisection algorithm to find dExcess_Up
    offset = np.full(n_on.shape, n_sig*n_sig*0.5)
    signal = excess0.copy() + 1
    ll = np.zeros(n_on.shape)
    for ii in range(len(ll)):
        ll[ii] = lm_loglikelihood(n_on[ii], n_off[ii], alpha[ii], signal[ii])
        count = 0
        while (l0[ii]-ll[ii]) < offset[ii] and np.isfinite(ll[ii]) and count < 100:
            signal[ii] = (signal[ii]-excess0[ii]) * 2. + excess0[ii]
            ll[ii] = lm_loglikelihood(n_on[ii], n_off[ii], alpha[ii], signal[ii])
            count += 1

    # Dichotomic search (i.e. interval bisection algorithm)
    low = excess0.copy()
    up = signal.copy()
    signal = (up+low)/2.
    for ii in range(len(ll)):
        ll[ii] = lm_loglikelihood(n_on[ii], n_off[ii], alpha[ii], signal[ii])
        count = 0.
        while np.abs(l0[ii]-ll[ii]-offset[ii]) > 1e-4 and np.isfinite(ll[ii]) and count < 100:
            if (l0[ii]-ll[ii]) > offset[ii]:
                up[ii] = signal[ii]
            else:
                low[ii] = signal[ii]
            signal[ii] = (up[ii]+low[ii])/2.
            ll[ii] = lm_loglikelihood(n_on[ii], n_off[ii], alpha[ii], signal[ii])
            count += 1

    if scalar:
        return np.abs(signal[0] - excess0[0])
    return np.abs(signal - excess0)


def lm_dexcess(n_on, n_off, alpha, n_sig=1.):
    r"""Compute mean excess uncertainties of an on-off observation based on Li&Ma [1].

    lm_dexcess = (lm_dexcess_up + lm_dexcess_down) / 2

    Parameters
    ----------
    n_on : array_like
        Observed number of counts in the on region
    n_off : array_like
        Observed number of counts in the off region
    alpha : array_like
        On / off region exposure ratio for background events
    n_sig : float
        Number of sigma

    Returns
    -------
    significance : `~numpy.ndarray`
        Error estimate

    References
    ----------
    .. [1] Li and Ma, "Analysis methods for results in gamma-ray astronomy",
       `Link <https://ui.adsabs.harvard.edu/abs/1983ApJ...272..317L>`_
    """
    err_up = lm_dexcess_up(n_on, n_off, alpha, n_sig)
    err_down = lm_dexcess_down(n_on, n_off, alpha, n_sig)
    err = (err_up + err_down) / 2.
    return err


def lm_significance_on_off(n_on, n_off, alpha):
    r"""
    Compute significance with the Li & Ma formula (17) [1]_.

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
    significance : `~numpy.ndarray`
        Significance estimate

    References
    ----------
    .. [1] Li and Ma, "Analysis methods for results in gamma-ray astronomy",
       `Link <https://ui.adsabs.harvard.edu/abs/1983ApJ...272..317L>`_
    """
    n_on = np.asanyarray(n_on, dtype=np.float64)
    n_off = np.asanyarray(n_off, dtype=np.float64)
    alpha = np.asanyarray(alpha, dtype=np.float64)

    sign = np.sign(n_on - alpha*n_off)

    tt = (alpha + 1) / (n_on + n_off)
    ll = n_on * np.log(n_on * tt / alpha)
    mm = n_off * np.log(n_off * tt)
    val = np.sqrt(np.abs(2 * (ll + mm)))

    return sign * val


def lm_significance(n_on, mu_bkg):
    r"""Compute significance for an observed number of counts and known background.

    IT gives the significance estimate corresponding
    to equation (17) from the Li & Ma paper [1]_ in the limiting of known background
    :math:`\mu_{bkg} = \alpha \times n_{off}` with :math:`\alpha \to 0`.

    It is given by the following formula:

    .. math::
        S_{lima} = \sqrt{2} \left[
          n_{on} \log \left( \frac{n_{on}}{\mu_{bkg}} \right) - n_{on} + \mu_{bkg}
        \right] ^ {1/2}

   Parameters
    ----------
    n_on : array_like
        Observed number of counts
    mu_bkg : array_like
        Known background level

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
    """
    sign = np.sign(n_on - mu_bkg)
    val = np.sqrt(2) * np.sqrt(n_on * np.log(n_on / mu_bkg) - n_on + mu_bkg)
    return sign * val
