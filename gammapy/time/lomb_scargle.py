from collections import OrderedDict
import numpy as np
from astropy.stats import LombScargle

__all__ = [
    'lomb_scargle',
    'lomb_scargle_plot',
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
    max_period = (np.max(time) - np.min(time)) / 10  # see Halpern ???
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
    This function computes the significance of periodogram peaks under certain criteria.
    To compute the Lomb-Scargle power spectral density,
    `~astropy.stats.LombScargle` is called.
    For eyesight inspection, the spectral window function is also returned to evaluate the impcat of sampling on the periodogram.

    More high-level docs. Link to a paper or wikipedia?

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
        - ``pre`` for pre-defined beta distribution (see Schwarzenberg-Czerny 1998)
        - ``cvm`` for Cramer-von-Mises distance minimisation (see Thieler et at. 2016)
        - ``nll`` for negative logarithmic likelihood minimisation
        - ``boot`` for bootstrap-resampling (see Sueveges 2012 and `astroML.time_series.lomb_scargle_bootstrap`.
    n_bootstraps : `float`
        Number of bootstraps resampling
        
    Returns
    -------
    results : `~collections.OrderedDict`
        Results dictionary (see description above).
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


def lomb_scargle_plot(time, flux, flux_error, freq, psd_data, best_period, significance, psd_win):
    """
    This function plots a light curve, its periodogram and spectral window function.
    The highest period of the periodogram and its significance will be added to the plot.

    Parameters
    ----------
    time : `~numpy.ndarray`
        Time array of the light curve
    flux : `~numpy.ndarray`
        Flux array of the light curve
    flux_err : `~numpy.ndarray`
        Flux error array of the light curve
    freq : `~numpy.ndarray`
        Frequencies for the periodogram
    psd_data : `~numpy.ndarray`
        Periodogram peaks of the data
    best_period : `float`
        Highest period of the periodogram
    significance : `float` or `~numpy.ndarray`
        Significance of ``best_period`` under the specified significance criterion. If the significance criterion is not defined, the maximum significance of all significance criteria is used
    psd_win : Periodogram peaks of the window function
        
    Returns
    -------
    plot
    """
    from matplotlib import gridspec
    from matplotlib import rc
    import matplotlib.pyplot as plt
    # define layout
    rc('text', usetex=True)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    # set up the figure & axes for plotting
    fig = plt.figure(figsize=(16, 9))    
    gs = gridspec.GridSpec(3, 1)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    ax3 = fig.add_subplot(gs[2, :])
    # plot the light curve
    ax1.errorbar(time, flux, flux_error, fmt='ok', elinewidth=1.5, capsize=0)
    ax1.set(xlabel=r'\textbf{time} (d)',
            ylabel=r'\textbf{magnitude} (a.u.)')
    # plot the periodogram
    ax2.plot(1. / freq, psd_data)
    # mark the best period and label with significance
    if np.isfinite(best_period):
        ax2.axvline(best_period, ymin=0, ymax=psd_data[freq == 1./best_period],
                    label=r'Detected period p = {:.1f} with {:.2f} significance'.format(best_period, np.max(significance)))
    ax2.set(  # xlabel=r'\textbf{period} (d)'
            ylabel=r'\textbf{power}',
            xlim=(0, np.max(1. / freq)),
            ylim=(0, 1),
    )
    ax2.legend(loc='upper right')
    # plot the spectral window function
    ax3.plot(1. / freq, psd_win)
    ax3.set(xlabel=r'\textbf{period} (d)',
            ylabel=r'\textbf{power}',
            xlim=(0, np.max(1. / freq)),
    )
