import numpy as np
from matplotlib import gridspec
from matplotlib import rc
import matplotlib.pyplot as plt

__all__ = [
    'plot_periodogram',
]


def plot_periodogram(time, flux, flux_error, periods, psd_data, psd_win, best_period='None', significance='None'):
    """
    Plot a light curve, its periodogram and spectral window function.
    The highest period of the periodogram and its significance will be added to the plot, if given.
    If multiple sginificance are forwarded, the lowest one will be used.

    Parameters
    ----------
    time : `~numpy.ndarray`
        Time array of the light curve
    flux : `~numpy.ndarray`
        Flux array of the light curve
    flux_error : `~numpy.ndarray`
        Flux error array of the light curve
    periods : `~numpy.ndarray`
        Periods for the periodogram
    psd_data : `~numpy.ndarray`
        Periodogram peaks of the data
    best_period : `float`
        Highest period of the periodogram
    significance : `float` or `~numpy.ndarray`
        Significance of ``best_period`` under the specified significance criterion.
        If the significance criterion is not defined, the maximum significance of all significance criteria is used
    psd_win : Periodogram peaks of the window function

    Returns
    -------
    plot
    """
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
    ax2.plot(periods, psd_data)
    # mark the best period and label with significance
    if best_period != 'None':
        # set precision for period format
        pre = int(abs(np.floor(np.log10(np.max(np.diff(periods))))))
        ax2.axvline(best_period, ymin=0, ymax=psd_data[periods == best_period],
                    label=r'Detected period p = {:.{}f} with {:.2f} significance'.format(best_period, pre, np.min(
                        list(significance.values()))))
    ax2.set(  # xlabel=r'\textbf{period} (d)'
        ylabel=r'\textbf{power}',
        xlim=(0, np.max(periods)),
        ylim=(0, 1),
    )
    ax2.legend(loc='upper right')
    # plot the spectral window function
    ax3.plot(periods, psd_win)
    ax3.set(xlabel=r'\textbf{period} (d)',
            ylabel=r'\textbf{power}',
            xlim=(0, np.max(periods)),
            )
    plt.savefig('example', bbox_inches='tight')
