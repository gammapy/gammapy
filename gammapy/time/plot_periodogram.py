import numpy as np
from matplotlib import gridspec
from matplotlib import rc
import matplotlib.pyplot as plt

__all__ = [
    'plot_periodogram',
]

def plot_periodogram(time, flux, flux_error, freq, psd_data, psd_win, best_period='None', significance='None'):
    """
    This function plots a light curve, its periodogram and spectral window function.
    The highest period of the periodogram and its significance will be added to the plot, if given.

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
    if best_period != 'None':
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
