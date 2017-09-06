# coding=utf-8
from collections import OrderedDict
import numpy as np
from scipy.optimize import least_squares

__all__ = [
    'robust_periodogram',
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
    time_win = np.rint(time / dt) * dt
    t_max = np.max(time_win)
    t_min = np.min(time_win)
    window_grid = np.arange(t_min, t_max+dt, dt)
    window_grid = np.rint(window_grid / dt) * dt  # round again since np.arange is not robust
    window = np.zeros(len(window_grid))
    window[np.searchsorted(window_grid, time_win, side='right')-1] = 1

    return window_grid, window


def _bootstrap(time, flux, flux_error, periods, n_bootstraps, loss, scale):
    """
    Returns value of maximum periodogram peak for every bootstrap resampling
    """
    rand = np.random.RandomState(42)
    if n_bootstraps == 'None':
        n_bootstraps = 100    
    max_periods = np.empty(n_bootstraps)
    second_max_periods = np.empty(n_bootstraps)
    beta0 = np.array([0, 1, 0])
    mu = np.median(flux)
    x = np.ones([len(time), len(beta0)])
    chi_model = np.empty([len(periods)])
    chi_noise = np.empty([len(periods)])
    for idx_run in range(n_bootstraps):
        ind = rand.randint(0, len(flux), len(flux))
        for i in range(len(periods)):
            chi_model[i] = least_squares(_full_model, beta0, loss=loss, f_scale=scale,
                                         args=(x, periods[i], time, flux[ind], flux_error[ind])).cost
            chi_noise[i] = least_squares(_location_model, mu, loss=loss, f_scale=scale,
                                         args=(time, flux[ind], flux_error[ind])).cost
        psd_boot = 1 - chi_model / chi_noise
        max_periods[idx_run] = np.max(psd_boot)
        second_max_periods[idx_run] = np.sort(psd_boot.flatten())[-2]

    return max_periods, second_max_periods


def _freq_grid(time, dt, max_period):
    """
    Generates the frequency grid for the periodogram
    """
    if dt == 'None':
        dt = (np.max(time) - np.min(time)) / (2 * len(time))
    if max_period == 'None':
        max_period = np.rint((np.max(time) - np.min(time)) / dt) * dt
    else:
        max_period = np.rint(max_period / dt) * dt
    min_period = dt
    periods = np.arange(min_period, max_period+dt, dt)
    grid = 1. / periods

    return grid, periods


def _significance_pre(time, periods, psd, psd_best_period):
    """
    Computes significance for the pre-defined beta distribution
    """
    from scipy.stats import beta
    a = (3-1)/2
    b = (len(time)-3)/2
    significance = 100 * beta.cdf(psd_best_period, a, b)**len(periods)
    print('pre')
    print(significance)
    print(beta.ppf(0.95 ** (1. / len(periods)), a, b))
    print(100 * beta.cdf(np.sort(psd.flatten())[-2], a, b)**len(periods))

    return significance


def _significance_cvm(periods, psd, psd_best_period):
    """
    Computes significance for the cvm-distance-minimised beta distribution
    """
    from scipy import optimize
    from scipy.stats import beta
    clip = 0.00001
    theta_1 = -1. * (np.mean(psd) * (-np.mean(psd)
                                     + np.mean(psd)**2
                                     + np.var(psd))
                     ) / (np.var(psd))
    if theta_1 < 0:
        theta_1 = clip
    theta_2 = (theta_1 - theta_1 * np.mean(psd)) / np.mean(psd)
    if theta_2 < 0:
        theta_2 = clip
    cvm_minimize = optimize.fmin(_cvm, [theta_1, theta_2], args=(psd,))
    significance = 100 * beta.cdf(psd_best_period, cvm_minimize[0], cvm_minimize[1])**len(periods)
    print('cvm')
    print(significance)
    print(beta.ppf(0.95 ** (1. / len(periods)), cvm_minimize[0], cvm_minimize[1]))
    print(100 * beta.cdf(np.sort(psd.flatten())[-2], cvm_minimize[0], cvm_minimize[1])**len(periods))

    return significance


def _significance_nll(time, periods, psd, psd_best_period):
    """
    Computes significance for the negative logarithmic likelihood-minimised beta distribution
    """
    from scipy import optimize
    from scipy.stats import beta
    a = (3-1)/2
    b = (len(time)-3)/2
    nll_minimize = optimize.fmin(_nll, [a, b], args=(psd,))
    significance = 100 * beta.cdf(psd_best_period, nll_minimize[0], nll_minimize[1])**len(periods)
    print('nll')
    print(significance)
    print(beta.ppf(0.95**(1./len(periods)), nll_minimize[0], nll_minimize[1]))
    print(100 * beta.cdf(np.sort(psd.flatten())[-2], nll_minimize[0], nll_minimize[1])**len(periods))

    return significance


def _significance_boot(time, flux, flux_error, periods, psd_best_period, n_bootstraps, loss, scale):
    """
    Computes significance for the bootstrap-resampling
    """
    from scipy import stats
    max_periods, second_max_periods = _bootstrap(time, flux, flux_error, periods, n_bootstraps, loss, scale)
    significance = 100 * (stats.percentileofscore(max_periods, psd_best_period) / 100)**len(periods)
    print('boot')
    print(significance)
    print(np.percentile(max_periods, 100 * 0.95**(1./len(periods))))
    print(100 * (stats.percentileofscore(second_max_periods, psd_best_period) / 100)**len(periods))

    return significance


def _significance_all(time, flux, flux_error, periods, psd, psd_best_period, n_bootstraps, loss, scale):
    """
    Computes significance for all significance criteria
    """
    significance = np.empty([4])
    significance[0] = _significance_pre(time, periods, psd, psd_best_period)
    significance[1] = _significance_nll(time, periods, psd, psd_best_period)
    significance[2] = _significance_cvm(periods, psd, psd_best_period)
    significance[3] = _significance_boot(time, flux, flux_error, periods, psd_best_period, n_bootstraps, loss, scale)

    return significance


def _full_model(beta0, x, period, t, y, dy):
    x[:, 1] = np.cos(2 * np.pi * t / period)
    x[:, 2] = np.sin(2 * np.pi * t / period)
    return (y - np.dot(x, beta0.T)) / dy


def _location_model(mu, t, y, dy):
    return (mu*np.ones(len(t)) - y) / dy


def robust_periodogram(time, flux, flux_err, dt, loss, scale, max_period='None', criteria='None', n_bootstraps='None'):
    """
    Compute period and significance of a light curve using robust regression techniques to fit a single harmonic model .

    To compute the power spectral density, the scipy object `~scipy.optimize.least_squares` is called.
    For eyesight inspection, the spectral window function is also returned to evaluate
    the impact of sampling on the periodogram.
    The significance cirteria are both, parametric and non-parametric.

    For an introduction to robust regression techniques
    and loss functions provided by scipy, see Nikolay Mayorov (2015).

    The function returns a results dictionary with the following content:

    - ``pgrid`` (`~numpy.ndarray`) -- Period grid in units of ``t``
    - ``psd`` (`~numpy.ndarray`) -- PSD of Lomb-Scargle at frequencies of ``fgrid``
    - ``period`` (`float`) -- Location of the highest periodogram peak
    - ``significance`` (`float`) or (`~numpy.ndarray`) -- Significance of ``period`` under specified criteria. 
      If criteria is not defined, the significance of all criteria is returned.
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
    loss : `str`
        loss function for the robust regression
        Available: `{'linear', 'soft_l1', 'huber', 'cauchy', 'arctan'}`
    scale : `float`
        loss scale parameter to define margin between inlier and outlier residuals
    max_period : `float`
        maximum period to analyse
    criteria : `list of str`
        Select which significance methods you'd like to run (by default all are running)
        Available: `{'pre', 'cvm', 'nll', 'boot'}`

        - ``pre`` for pre-defined beta distribution (see Schwarzenberg-Czerny (1998))
        - ``cvm`` for Cramer-von-Mises distance minimisation (see Thieler et at. (2016))
        - ``nll`` for negative logarithmic likelihood minimisation
        - ``boot`` for bootstrap-resampling (see Süveges (2012) and
        `astroML.time_series.lomb_scargle_bootstrap <http://www.astroml.org/modules/generated/astroML.time_series.lomb_scargle_bootstrap.html>`_)
    
    n_bootstraps : `float`
        Number of bootstraps resampling
        
    Returns
    -------
    results : `~collections.OrderedDict`
        Results dictionary (see description above).

    References
    ----------
    .. [1] Nikolay Mayorov (2015), `Link <http://scipy-cookbook.readthedocs.io/items/robust_regression.html>`_
    .. [2] Schwarzenberg-Czerny (1998), "The distribution of empirical periodograms: Lomb-Scargle and PDM spectra",
       `Link <https://academic.oup.com/mnras/article/301/3/831/1038387/The-distribution-of-empirical-periodograms-Lomb>`_
    .. [3] Thieler et at. (2016), "RobPer: An R Package to Calculate Periodograms for Light Curves Based on Robust Regression",
       `Link <https://www.jstatsoft.org/article/view/v069i09>`_
    .. [4] Süveges (2012), "False Alarm Probability based on bootstrap and extreme-value methods for periodogram peaks",
       `Link <https://www.researchgate.net/profile/Maria_Sueveges/publication/267988824_False_Alarm_Probability_based_on_bootstrap_and_extreme-value_methods_for_periodogram_peaks/links/54e1ba3a0cf2953c22bb222a.pdf>`_
    """

    # set up period grid
    freq, periods = _freq_grid(time, dt, max_period)

    # set up model for robust regression
    beta0 = np.array([0, 1, 0])
    mu = np.median(flux)
    x = np.ones([len(time), len(beta0)])
    chi_model = np.empty([len(periods)])
    chi_noise = np.empty([len(periods)])

    # perform robust regression on light curve
    for i in range(len(periods)):
        chi_model[i] = least_squares(_full_model, beta0, loss=loss, f_scale=scale,
                                     args=(x, periods[i], time, flux, flux_err)).cost
        chi_noise[i] = least_squares(_location_model, mu, loss=loss, f_scale=scale,
                                     args=(time, flux, flux_err)).cost
    psd_data = 1 - chi_model / chi_noise

    # find period with highest periodogram peak
    psd_best_period = np.max(psd_data)
    best_period = periods[np.argmax(psd_data)]
    print(best_period)

    # define significance for best period
    if criteria == 'None':
        criteria = ['pre', 'cvm', 'nll', 'boot']

    significance = OrderedDict()

    if 'pre' in criteria:
        significance['pre'] = _significance_pre(time, periods, psd_data, psd_best_period)
    if 'cvm' in criteria:
        significance['cvm'] = _significance_cvm(periods, psd_data, psd_best_period)
    if 'nll' in criteria:
        significance['nll'] = _significance_nll(time, periods, psd_data, psd_best_period)
    if 'boot' in criteria:
        significance['boot'] = _significance_boot(time, flux, flux_err, periods, psd_best_period,
                                                  n_bootstraps, loss, scale)

    # spectral window function
    print('window')
    time_win, window = _window_function(time, dt)
    x = np.ones([len(time_win), len(beta0)])
    for i in range(len(periods)):
        print(i)
        chi_model[i] = least_squares(_full_model, beta0, loss='linear', f_scale=scale,
                                     args=(x, periods[i], time_win, window, 1)).cost
        chi_noise[i] = least_squares(_location_model, mu, loss='linear', f_scale=scale,
                                     args=(time_win, window, 1)).cost
    psd_win = 1 - chi_model / chi_noise

    return OrderedDict([
        ('pgrid', periods),
        ('psd', psd_data),
        ('period', best_period),
        ('significance', significance),
        ('swf', psd_win),
    ])
