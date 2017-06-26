from collections import OrderedDict
import numpy as np
from astropy.stats import LombScargle

__all__ = [
    'lomb_scargle',
]


def _nll(param, data):
    """
    Negative log likelihood function for beta distribution
    """
    from scipy.stats import beta
    a, b = param
    pdf = beta.pdf(data, a, b, loc=0, scale=1)
    lg = np.log(pdf)
    mask = np.isfinite(lg)
    nll = -lg[mask].sum()
    return nll


def _cvm(param, data):
    """
    Cramer-von-Mises distance for beta distribution
    """
    from scipy.stats import beta
    a, b = param
    ordered_data = np.sort(data, axis=None)
    for n in range(len(data)):
        sumbeta =+ (beta.cdf(ordered_data[n],
                             a, b,
                             loc=0, scale=1)
                    - (n - 0.5) / len(data))**2.0
    cvm_dist = (1. / len(data)) * sumbeta + 1. / (12 * (len(data)**2.))
    mask = np.isfinite(cvm_dist)
    cvm = cvm_dist[mask]
    return cvm


def _window_function(time, dt):
    """
    Generates window function with desired resolution dt
    """
    n_points = (np.max(time) - np.min(time)) / dt
    t_win = np.linspace(np.min(time), np.max(time), n_points+1, endpoint=True)
    window = np.zeros(len(t_win))
    window[np.searchsorted(t_win, time)] = 1

    return t_win, window


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


def _freq_grid(time, dt):
    """
    Generates the frequency grid for the periodogram
    """
    max_period = (np.max(time) - np.min(time))
    min_period = dt
    n_periods = max_period / min_period
    periods = np.linspace(min_period, max_period, n_periods)
    grid = 1. / periods

    return grid


def _significance_pre(time, freq, psd_best_period):
    """
    Computes significance for the pre-defined beta distribution
    """
    from scipy.stats import beta
    a = (3-1)/2
    b = (len(time)-3)/2
    significance = 100 * beta.cdf(psd_best_period, a, b)**len(freq)

    return significance


def _significance_cvm(time, freq, psd, psd_best_period):
    """
    Computes significance for the cvm-distance-minimised beta distribution
    """
    from scipy import optimize
    from scipy.stats import beta
    theta_1 = -1. * (np.mean(psd) * (-np.mean(psd)
                                     + np.mean(psd)**2
                                     + np.var(psd))
                     ) / (np.var(psd))
    if theta_1 < 0:
        theta_1 = 0.00001
    theta_2 = (theta_1 - theta_1 * np.mean(psd)) / np.mean(psd)
    if theta_2 < 0:
        theta_2 = 0.00001
    cvm_minimize = optimize.fmin(_cvm, [theta_1, theta_2], args=(psd,))
    significance = 100 * beta.cdf(psd_best_period, cvm_minimize[0], cvm_minimize[1])**len(freq)

    return significance


def _significance_nll(time, freq, psd, psd_best_period):
    """
    Computes significance for the negative logarithmic likelihood-minimised beta distribution
    """
    from scipy import optimize
    from scipy.stats import beta
    a = (3-1)/2
    b = (len(time)-3)/2
    nll_minimize = optimize.fmin(_nll, [a, b], args=(psd,))
    significance = 100 * beta.cdf(psd_best_period, nll_minimize[0], nll_minimize[1])**len(freq)

    return significance


def _significance_boot(time, flux, flux_error, freq, psd_best_period, n_bootstraps):
    """
    Computes significance for the bootstrap-resampling
    """
    from scipy import stats
    max_periods = _bootstrap(time, flux, flux_error, freq, n_bootstraps)
    significance = stats.percentileofscore(max_periods, psd_best_period)

    return significance


def _significance_all(time, flux, flux_error, freq, psd, psd_best_period, n_bootstraps):
    """
    Computes significance for all significance criteria
    """
    significance = np.empty([4])
    significance[0] = _significance_pre(time, freq, psd_best_period)
    significance[1] = _significance_nll(time, freq, psd, psd_best_period)
    significance[2] = _significance_cvm(time, freq, psd, psd_best_period)
    significance[3] = _significance_boot(time, flux, flux_error, freq, psd_best_period, n_bootstraps=100)

    return significance


def lomb_scargle(time, flux, flux_error, dt, criterion='None', n_bootstraps=100):
    """
    This function computes the significance of periodogram peaks under certain significance criteria.
    To compute the Lomb-Scargle power spectral density, the astropy object `~astropy.stats.LombScargle` is called.
    For eyesight inspection, the spectral window function is also returned to evaluate the impcat of sampling on the periodogram.

    For an introduction to the Lomb-Scargle periodogram, see Lomb (1976) and Scargle (1982).

    The function returns a results dictionary with the following content:

    - ``fgrid`` (`~numpy.ndarray`) -- Frequency grid in inverse units of ``t``
    - ``psd`` (`~numpy.ndarray`) -- Power spectral density of the Lomb-Scargle periodogram at the frequencies of ``fgrid``
    - ``period`` (`float`) -- Location of the highest periodogram peak
    - ``significance`` (`float`) or (`~numpy.ndarray`) -- Significance of ``period`` under the specified significance criterion. If the significance criterion is not defined, all significance criteria are used and their respective significance for the period is returned
    - ``swf`` (`~numpy.ndarray`) -- Spectral window function

    Parameters
    ----------
    time : `~numpy.ndarray`
        Time array of the light curve
    flux : `~numpy.ndarray`
        Flux array of the light curve
    flux_err : `~numpy.ndarray`
        Flux error array of the light curve
    dt : `float`
        desired resolution of the periodogram and the window function
    criterion : `string`
        significance criterion

        - ``pre`` for pre-defined beta distribution (see Schwarzenberg-Czerny (1998))
        - ``cvm`` for Cramer-von-Mises distance minimisation (see Thieler et at. (2016))
        - ``nll`` for negative logarithmic likelihood minimisation
        - ``boot`` for bootstrap-resampling (see Süveges (2012) and `astroML.time_series.lomb_scargle_bootstrap <http://www.astroml.org/modules/generated/astroML.time_series.lomb_scargle_bootstrap.html>`_)
    
    n_bootstraps : `float`
        Number of bootstraps resampling
        
    Returns
    -------
    results : `~collections.OrderedDict`
        Results dictionary (see description above).

    References
    ----------
    .. [1] Lomb (1976), "Least-squares frequency analysis of unequally spaced data", 
       `Link <https://link.springer.com/article/10.1007%2FBF00648343?LI=true>`_
    .. [2] Scargle (1982), "Studies in astronomical time series analysis. II - Statistical aspects of spectral analysis of unevenly spaced data",
       `Link <http://articles.adsabs.harvard.edu/full/1982ApJ...263..835S>`_
    .. [3] Schwarzenberg-Czerny (1998), "The distribution of empirical periodograms: Lomb-Scargle and PDM spectra",
       `Link <https://academic.oup.com/mnras/article/301/3/831/1038387/The-distribution-of-empirical-periodograms-Lomb>`_
    .. [4] Thieler et at. (2016), "RobPer: An R Package to Calculate Periodograms for Light Curves Based on Robust Regression",
       `Link <https://www.jstatsoft.org/article/view/v069i09>`_
    .. [5] Süveges (2012), "False Alarm Probability based on bootstrap and extreme-value methods for periodogram peaks",
       `Link <https://www.researchgate.net/profile/Maria_Sueveges/publication/267988824_False_Alarm_Probability_based_on_bootstrap_and_extreme-value_methods_for_periodogram_peaks/links/54e1ba3a0cf2953c22bb222a.pdf>`_
    """
    # set up lomb-scargle-algorithm
    freq = _freq_grid(time, dt)
    psd_data = LombScargle(time, flux, flux_error).power(freq)

    # find period with highest periodogram peak
    psd_best_period = np.max(psd_data)
    best_period = 1. / freq[np.argmax(psd_data)]

    # define significance for best period
    if criterion == 'pre':
        significance = _significance_pre(time, freq, psd_best_period)
    if criterion == 'cvm':
        significance = _significance_cvm(time, freq, psd_data, psd_best_period)
    if criterion == 'nll':
        significance = _significance_nll(time, freq, psd_data, psd_best_period)
    if criterion == 'boot':
        significance = _significance_boot(time, flux, flux_error, freq, psd_best_period, n_bootstraps)
    if criterion == 'None':
        significance = _significance_all(time, flux, flux_error, freq, psd_data, psd_best_period, n_bootstraps)

    # spectral window function
    t_win, window = _window_function(time, dt)
    psd_win = LombScargle(t_win, window, 1).power(freq)

    return OrderedDict([
        ('fgrid', freq),
        ('psd', psd_data),
        ('period', best_period),
        ('significance', significance),
        ('swf', psd_win),
    ])

