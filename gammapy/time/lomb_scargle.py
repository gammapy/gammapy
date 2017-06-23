from collections import OrderedDict
import numpy as np
from astropy.stats import LombScargle

__all__ = [
    'lomb_scargle',
    'lomb_scargle_plot',
]


def _nll(param, data):
    """
    Negative log likelihood function for beta
    <param>: list for parameters to be fitted.
    <args>: 1-element array containing the sample data.
    Return <nll>: negative log-likelihood to be minimized.
    """
    from scipy.stats import beta
    a, b = param
    pdf = beta.pdf(data, a, b, loc=0, scale=1)
    lg = np.log(pdf)
    mask = np.isfinite(lg)
    nll = -lg[mask].sum()
    return nll


def _cvm(param, data):
    """Cramer-von-Mises distance minimization
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
    # TODO: this needs explanation what it does
    # TODO: rewrite in better way possible?
    # Remove? Punt this to the user, help a little with a docs example?
    # Alternative: extra utility function that user can call to compute dt before calling the bootstrap function below.
    n_points = (np.max(time) - np.min(time)) / dt
    t_win = np.linspace(np.min(time), np.max(time), n_points+1, endpoint=True)
    window = np.zeros(len(t_win))
    window[np.searchsorted(t_win, time)] = 1

    return t_win, window


def _bootstrap(time, flux, flux_error, freq, n_bootstraps):
    # tbd: does the caller need control over seed / bootstrap options?
    rand = np.random.RandomState(42)
    max_periods = np.empty(n_bootstraps)
    for idx_run in range(n_bootstraps):
        ind = rand.randint(0, len(flux), len(flux))
        psd_boot = LombScargle(time, flux[ind], flux_error[ind]).power(freq)
        max_periods[idx_run] = np.max(psd_boot)

    return max_periods


def _freq_grid(time, dt):
    # What is this? -> rename to _freq_grid?
    max_period = (np.max(time) - np.min(time)) / 10  # see Halpern ???
    min_period = dt
    n_periods = max_period / min_period
    periods = np.linspace(min_period, max_period, n_periods)
    grid = 1. / periods

    return grid


def _significance_pre(time, freq, psd_best_period):
    from scipy.stats import beta
    a = (3-1)/2
    b = (len(time)-3)/2
    significance = 100 * beta.cdf(psd_best_period, a, b)**len(freq)

    return significance


def _significance_cvm(time, freq, psd, psd_best_period):
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
    from scipy import optimize
    from scipy.stats import beta
    a = (3-1)/2
    b = (len(time)-3)/2
    nll_minimize = optimize.fmin(_nll, [a, b], args=(psd,))
    significance = 100 * beta.cdf(psd_best_period, nll_minimize[0], nll_minimize[1])**len(freq)

    return significance


def _significance_boot(time, flux, flux_error, freq, psd_best_period, n_bootstraps):
    from scipy import stats
    max_periods = _bootstrap(time, flux, flux_error, freq, n_bootstraps)
    significance = stats.percentileofscore(max_periods, psd_best_period)

    return significance


def _significance_all(time, flux, flux_error, freq, psd, psd_best_period, n_bootstraps):
    significance = np.empty([4])
    significance[0] = _significance_pre(time, freq, psd_best_period)
    significance[1] = _significance_nll(time, freq, psd, psd_best_period)
    significance[2] = _significance_cvm(time, freq, psd, psd_best_period)
    significance[3] = _significance_boot(time, freq, psd_best_period)

    return significance


def _def_significance(freq, psd, quantile):
    # TODO: split this out into a utility function?
    if np.max(psd) > quantile:
        best_period = freq[np.argmax(psd)]
    else:
        best_period = np.nan

    return best_period


def lomb_scargle(time, flux, flux_error, dt, criterion, n_bootstraps='None'):
    """Processs the lomb-scargle-algorithmus in autopower mode.

    More high-level docs. Link to a paper or wikipedia?

    This function computes significance bla bla.
    To compute the Lomb-Scargle power spectral density,
    `~astropy.stats.LombScargle` is called.

    The function returns a results dictionary with the following conent:

    - ``fgrid`` (`~numpy.ndarray`) -- Frequency grid in inverse units of ``t``
    - ``PSD`` (array) -- power spectral density of the Lomb-Scargle periodogram at the frequencies of ``fgrid``

    Parameters
    ----------
    time : `~numpy.ndarray`
        Time
    flux : `~numpy.ndarray`
        Flux
    flux_err : `~numpy.ndarray`
        Flux error

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
    if criterion == 'all':
        significance = _significance_all(time, flux, flux_error, freq, psd, psd_best_period, n_bootstraps)

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


def lomb_scargle_plot(time, flux, flux_error, freq, psd, best_period, significance, psd_win):
    """TODO: describe

    Parameters
    ----------
    bla
    """
    from matplotlib import gridspec
    from matplotlib import rc
    import matplotlib.pyplot as plt
    # plotting
    # set up the figure & axes for plotting
    fig = plt.figure(figsize=(16, 9))
    # rcParams['axes.labelsize'] = 16
    rc('text', usetex=True)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    gs = gridspec.GridSpec(3, 1)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    ax3 = fig.add_subplot(gs[2, :])
    # plot the raw data
    ax1.errorbar(time, flux, flux_error, fmt='ok', elinewidth=1.5, capsize=0)
    ax1.set(xlabel=r'\textbf{time} (d)',
            ylabel=r'\textbf{magnitude} (a.u.)')
    # plot the periodogram
    ax2.plot(1. / freq, psd)
    if np.isfinite(best_period):
        ax2.axvline(best_period, ymin=0, ymax=psd[freq == 1./best_period],
                    label=r'Detected period p = {:.1f} with {:.2f} significance'.format(best_period, significance))

    ax2.set(  # xlabel=r'\textbf{period} (d)'
            ylabel=r'\textbf{power}',
            xlim=(0, np.max(1. / freq)),
            ylim=(0, 1),
    )
    ax2.legend(loc='upper right')
    # plot the window function
    ax3.plot(1. / freq, psd_win)
    ax3.set(xlabel=r'\textbf{period} (d)',
            ylabel=r'\textbf{power}',
            xlim=(0, np.max(1. / freq)),
    )
