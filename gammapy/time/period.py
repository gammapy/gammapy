# Licensed under a 3-clause BSD style license - see LICENSE.rst
from collections import OrderedDict
import numpy as np
from astropy.stats import LombScargle

__all__ = [
    'lomb_scargle',
]


def lomb_scargle(time, flux, flux_err, dt, max_period=None, criteria='all', n_bootstraps=100):
    """
    Compute period and its false alarm probability of a light curve using Lomb-Scargle PSD.

    To compute the Lomb-Scargle power spectral density, `astropy.stats.LombScargle` is called.
    For eyesight inspection, the spectral window function is also returned
    to evaluate the impact of sampling on the periodogram.
    The criteria for the false alarm probability are both parametric and non-parametric.

    For an introduction to the Lomb-Scargle periodogram, see Lomb (1976) and Scargle (1982).
    For an introduction to the false alarm probability of thr Lomb-Scargle periodogram, see the astropy docs.

    The function returns a results dictionary with the following content:

    - ``pgrid`` (`~numpy.ndarray`) -- Period grid in units of ``t``
    - ``psd`` (`~numpy.ndarray`) -- PSD of Lomb-Scargle at frequencies of ``fgrid``
    - ``period`` (`float`) -- Location of the highest periodogram peak
    - ``fap`` (`float`) or (`~numpy.ndarray`) -- False alarm probability of ``period``
      under the null hypothesis of only-noise data for the specified criteria.
      If criteria is not defined, the false alarm probability of all criteria is returned.
    - ``swf`` (`~numpy.ndarray`) -- Spectral window function

    Parameters
    ----------
    time : `~numpy.ndarray`
        Time array of the light curve
    flux : `~numpy.ndarray`
        Flux array of the light curve
    flux_err : `~numpy.ndarray`
        Flux error array of the light curve
    dt : float
        Desired resolution of the periodogram and the window function
    max_period : float
        Maximum period to analyse
    criteria : list of str
        Select which significance methods you'd like to run (by default all are running)
        Available: `{'pre', 'cvm', 'nll', 'boot'}`

        - ``pre`` for pre-defined beta distribution (see Schwarzenberg-Czerny (1998))
        - ``cvm`` for Cramer-von-Mises distance minimisation (see Thieler et at. (2016))
        - ``nll`` for negative logarithmic likelihood minimisation
        - ``boot`` for bootstrap-resampling (see Sueveges (2012)
           and ``astroML.time_series.lomb_scargle_bootstrap``.
    n_bootstraps : int
        Number of bootstraps resampling

    Returns
    -------
    results : `~collections.OrderedDict`
        Results dictionary (see description above).

    References
    ----------
    .. [1] Lomb (1976), "Least-squares frequency analysis of unequally spaced data",
       `Link <https://link.springer.com/article/10.1007%2FBF00648343?LI=true>`__
    .. [2] Scargle (1982), "Studies in astronomical time series analysis. II -
       Statistical aspects of spectral analysis of unevenly spaced data",
       `Link <http://articles.adsabs.harvard.edu/full/1982ApJ...263..835S>`__
    .. [3] Schwarzenberg-Czerny (1998), "The distribution of empirical periodograms: Lomb-Scargle and PDM spectra",
       `Link <https://academic.oup.com/mnras/article/301/3/831/1038387/The-distribution-of-empirical-periodograms-Lomb>`__
    .. [4] Thieler et at. (2016), "RobPer: An R Package to Calculate Periodograms for Light Curves Based on Robust Regression",
       `Link <https://www.jstatsoft.org/article/view/v069i09>`__
    .. [5] Sueveges (2012), "False Alarm Probability based on bootstrap and extreme-value methods for periodogram peaks",
       `Link <http://ada7.cosmostat.org/ADA7_proceeding_MSuveges2.pdf>`__
    .. [6] Astropy docs, Lomb-Scargle Periodograms,
       `Link <http://docs.astropy.org/en/latest/stats/lombscargle.html>`_
    """
    # set up lomb-scargle-algorithm
    freq, periods = _freq_grid(time, dt, max_period)
    psd_data = LombScargle(time, flux, flux_err).power(freq)

    # find period with highest periodogram peak
    psd_best_period = np.max(psd_data)
    best_period = periods[np.argmax(psd_data)]

    # define significance for best period
    if criteria == 'all':
        criteria = ['pre', 'cvm', 'nll', 'boot']

    fap = OrderedDict()

    if 'pre' in criteria:
        fap['pre'] = _fap_pre(time, freq, psd_best_period)
    if 'cvm' in criteria:
        fap['cvm'] = _fap_cvm(freq, psd_data, psd_best_period)
    if 'nll' in criteria:
        fap['nll'] = _fap_nll(time, freq, psd_data, psd_best_period)
    if 'boot' in criteria:
        fap['boot'] = _fap_boot(time, flux, flux_err, freq, psd_best_period, n_bootstraps)

    # spectral window function
    time_win, window = _window_function(time, dt)
    psd_win = LombScargle(time_win, window, 1).power(freq)

    return OrderedDict([
        ('pgrid', periods),
        ('psd', psd_data),
        ('period', best_period),
        ('fap', fap),
        ('swf', psd_win),
    ])


def _window_function(time, dt):
    """
    Generates window function with desired resolution dt
    """
    time_win = np.rint(time / dt) * dt
    t_max = np.max(time_win)
    t_min = np.min(time_win)
    window_grid = np.arange(t_min, t_max + dt, dt)
    window_grid = np.rint(window_grid / dt) * dt  # round again since np.arange is not robust
    window = np.zeros(len(window_grid))
    window[np.searchsorted(window_grid, time_win, side='right') - 1] = 1

    return window_grid, window


def _freq_grid(time, dt, max_period):
    """
    Generates the frequency grid for the periodogram
    """
    if max_period is None:
        max_period = np.rint((np.max(time) - np.min(time)) / dt) * dt
    else:
        max_period = np.rint(max_period / dt) * dt

    min_period = dt
    periods = np.arange(min_period, max_period + dt, dt)
    grid = 1. / periods

    return grid, periods


def _cvm(param, data):
    """
    Cramer-von-Mises distance for beta distribution
    """
    from scipy.stats import beta
    a, b = param
    ordered_data = np.sort(data, axis=None)
    sumbeta = 0
    for n in range(len(data)):
        cdf = beta.cdf(ordered_data[n], a, b)
        sumbeta += (cdf - (n - 0.5) / len(data)) ** 2.0

    cvm_dist = (1. / len(data)) * sumbeta + 1. / (12 * (len(data) ** 2.))
    mask = np.isfinite(cvm_dist)
    cvm = cvm_dist[mask]

    return cvm


def _nll(param, data):
    """
    Negative log likelihood function for beta distribution
    """
    from scipy.stats import beta
    a, b = param
    pdf = beta.pdf(data, a, b)
    lg = np.log(pdf)
    mask = np.isfinite(lg)
    nll = -lg[mask].sum()

    return nll


def _bootstrap(time, flux, flux_error, freq, n_bootstraps):
    """
    Returns value of maximum periodogram peak for every bootstrap resampling
    """
    rand = np.random.RandomState(42)
    max_periods = np.empty(n_bootstraps)

    for idx_run in range(n_bootstraps):
        ind = rand.randint(0, len(flux), len(flux))
        psd_boot = LombScargle(time, flux[ind], flux_error[ind]).power(freq)
        max_periods[idx_run] = np.max(psd_boot)

    return max_periods


def _fap_pre(time, freq, psd_best_period):
    """
    Computes false alarm probability for the pre-defined beta distribution
    """
    from scipy.stats import beta
    a = (3 - 1) / 2
    b = (len(time) - 3) / 2
    fap = 1 - beta.cdf(psd_best_period, a, b) ** len(freq)

    return fap


def _fap_cvm(freq, psd, psd_best_period):
    """
    Computes false alarm probability for the cvm-distance-minimised beta distribution
    """
    from scipy import optimize
    from scipy.stats import beta
    clip = 0.00001

    temp = np.mean(psd) * (-np.mean(psd) + np.mean(psd) ** 2 + np.var(psd))
    theta_1 = - temp / np.var(psd)
    if theta_1 < 0:
        theta_1 = clip

    theta_2 = (theta_1 - theta_1 * np.mean(psd)) / np.mean(psd)
    if theta_2 < 0:
        theta_2 = clip

    cvm_minimize = optimize.minimize(
        _cvm,
        [theta_1, theta_2],
        args=(psd,),
        bounds=((clip, None), (clip, None)),
    )
    fap = 1 - beta.cdf(psd_best_period, cvm_minimize.x[0], cvm_minimize.x[1]) ** len(freq)

    return fap


def _fap_nll(time, freq, psd, psd_best_period):
    """
    Computes false alarm probability for the negative logarithmic likelihood-minimised beta distribution
    """
    from scipy import optimize
    from scipy.stats import beta
    a = (3 - 1) / 2
    b = (len(time) - 3) / 2
    clip = 0.00001
    nll_minimize = optimize.minimize(
        _nll,
        [a, b],
        args=(psd,),
        bounds=((clip, None), (clip, None)),
    )
    fap = 1 - beta.cdf(psd_best_period, nll_minimize.x[0], nll_minimize.x[1]) ** len(freq)

    return fap


def _fap_boot(time, flux, flux_error, freq, psd_best_period, n_bootstraps):
    """
    Computes significance for the bootstrap-resampling
    """
    from scipy import stats
    max_periods = _bootstrap(time, flux, flux_error, freq, n_bootstraps)
    fap = 1 - (stats.percentileofscore(max_periods, psd_best_period) / 100)

    return fap
