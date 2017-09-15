# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np

__all__ = [
    'plot_periodogram',
]


def plot_periodogram(time, flux, flux_err, periods, psd_data, psd_win, best_period=None, fap=None):
    """Plot a light curve, its periodogram and spectral window function.

    The highest period of the periodogram and its significance will be added to the plot, if given.
    If multiple significances are forwarded, the lowest one will be used.

    Parameters
    ----------
    time : `~numpy.ndarray`
        Time array of the light curve
    flux : `~numpy.ndarray`
        Flux array of the light curve
    flux_err : `~numpy.ndarray`
        Flux error array of the light curve
    periods : `~numpy.ndarray`
        Periods for the periodogram
    psd_data : `~numpy.ndarray`
        Periodogram peaks of the data
    psd_win : `~numpy.ndarray`
        Periodogram peaks of the window function
    best_period : float
        Highest period of the periodogram
    fap : float or `~numpy.ndarray`
        False alarm probability of ``best_period`` under the specified significance criterion.
        If the significance criterion is not defined, the maximum false alarm probability
        of all significance criteria is used.
        
    Returns
    -------
    fig : `~matplotlib.figure.Figure`
        Matplotlib figure
    """
    import matplotlib.pyplot as plt

    # set up the figure & axes for plotting
    fig = plt.figure(figsize=(16, 9))
    grid_spec = plt.GridSpec(3, 1)

    # plot the light curve
    ax = fig.add_subplot(grid_spec[0, :])
    ax.errorbar(time, flux, flux_err, fmt='ok', elinewidth=1.5, capsize=0)
    ax.set_xlabel('time (d)')
    ax.set_ylabel('magnitude (a.u.)')

    # plot the periodogram
    ax = fig.add_subplot(grid_spec[1, :])
    ax.plot(periods, psd_data)
    # mark the best period and label with significance
    if best_period is not None:
        if fap is None:
            raise ValueError('Must give a false alarm probability if you give a best_period')

        # set precision for period format
        pre = int(abs(np.floor(np.log10(np.max(np.diff(periods))))))
        fap_max = max(fap.values())
        label = 'Detected period p = {:.{}f} with {:.2E} FAP'.format(best_period, pre, fap_max)
        ymax = psd_data[periods == best_period]
        ax.axvline(best_period, ymin=0, ymax=ymax, label=label)

    ax.set_xlabel('period (d)')
    ax.set_ylabel('power')
    ax.set_xlim(0, np.max(periods))
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right')

    # plot the spectral window function
    ax = fig.add_subplot(grid_spec[2, :])
    ax.plot(periods, psd_win)
    ax.set_xlabel('period (d)')
    ax.set_ylabel('power')
    ax.set_xlim(0, np.max(periods))

    return fig
