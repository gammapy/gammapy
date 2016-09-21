# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.time import Time, TimeDelta
from ..utils.time import TIME_REF_FERMI

__all__ = [
    'plot_fermi_3fgl_light_curve',
]


def plot_time_difference_distribution(time, ax=None):
    """Plot event time difference distribution.

    Parameters
    ----------
    time : `~astropy.time.Time`
        Event times (must be sorted)
    ax : `~matplotlib.axes.Axes` or None
        Axes
    Returns
    -------
    ax : `~matplotlib.axes.Axes`
        Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gcf()

    td = time[1:] - time[:-1]

    # TODO: implement!
    raise NotImplementedError


def plot_fermi_3fgl_light_curve(source_name, time_start=None, time_end=None, ax=None):
    """Plot flux as a function of time for a fermi 3FGL object.

    Parameters
    ----------
    source_name : str
        The 3FGL catalog name of the object to plot
    time_start : `~astropy.time.Time` or str or None
        Light curve start time.  If None, use the earliest time in the catalog.
    time_end : `~astropy.time.Time` or str or None
        Light curve end time.  If None, use the latest time in the catalog.
    ax : `~matplotlib.axes.Axes` or None
        Axes

    Returns
    -------
    ax : `~matplotlib.axes.Axes`
        Axes

    Examples
    --------
    Plot a 3FGL lightcurve:

    .. plot::
        :include-source:

        from gammapy.time import plot_fermi_3fgl_light_curve
        plot_fermi_3fgl_light_curve('3FGL J0349.9-2102',
                                    time_start='2010-01-01',
                                    time_end='2015-02-02')

        import matplotlib.pyplot as plt
        plt.show()
    """
    from ..catalog import fetch_fermi_catalog
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    ax = plt.gca() if ax is None else ax

    if time_start is None:
        time_start = Time('2008-08-02T00:33:19')
    else:
        time_start = Time(time_start)

    if time_end is None:
        time_end = Time('2012-07-31T22:45:47')
    else:
        time_end = Time(time_end)

    fermi_met_start = (time_start - TIME_REF_FERMI).sec

    fermi_met_end = (time_end - TIME_REF_FERMI).sec

    fermi_cat = fetch_fermi_catalog('3FGL')

    catalog_index = np.where(fermi_cat[1].data['Source_Name'] == source_name)[0][0]

    hist_start = fermi_cat[3].data['Hist_Start']
    time_index_start = np.where(hist_start >= fermi_met_start)[0][0]

    # The final entry is the end of the last bin, so no off by one error
    time_index_end = np.where(hist_start <= fermi_met_end)[0][-1] + 1

    time_start = hist_start[time_index_start: time_index_end]
    time_end = np.roll(time_start, -1)

    time_diff = 0.5 * (time_end - time_start)

    # Trim because there is one more bin edge than there is bin mid point
    time_diff = time_diff[0:-1]

    # Midpoints of each bin.
    time_mid = time_start[0:-1] + time_diff

    cat_row = fermi_cat[1].data[catalog_index]

    flux_history = cat_row['Flux_History'][time_index_start: time_index_end]

    flux_history_lower_bound = cat_row['Unc_Flux_History'][time_index_start: time_index_end, 0]
    flux_history_upper_bound = cat_row['Unc_Flux_History'][time_index_start: time_index_end, 1]
    flux_history_lower_bound = abs(flux_history_lower_bound)

    time_mid = (TIME_REF_FERMI + TimeDelta(time_mid, format='sec'))

    time_at_bin_start = time_mid - TimeDelta(time_diff, format='sec')

    time_at_bin_end = time_mid + TimeDelta(time_diff, format='sec')

    time_mid = time_mid.plot_date

    time_at_bin_start = time_at_bin_start.plot_date

    time_at_bin_end = time_at_bin_end.plot_date

    time_diff_at_bin_start = time_mid - time_at_bin_start

    time_diff_at_bin_end = time_at_bin_end - time_mid

    # Where a lower bound was recorded.
    idx1 = np.where(np.invert(np.isnan(flux_history_lower_bound)))

    # Where a lower bound was not recorded.
    idx2 = np.where(np.isnan(flux_history_lower_bound))

    # Where no lower bound was recorded, set to zero flux.
    flux_history_lower_bound[idx2] = flux_history[idx2]

    # Plot data points and upper limits.
    ax.errorbar(time_mid[idx1], flux_history[idx1],
                yerr=(flux_history_lower_bound[idx1], flux_history_upper_bound[idx1]),
                xerr=(time_diff_at_bin_start[idx1], time_diff_at_bin_end[idx1]),
                marker='o', elinewidth=1, linewidth=0, color='black')
    ax.errorbar(time_mid[idx2], flux_history[idx2],
                yerr=(flux_history_lower_bound[idx2], flux_history_upper_bound[idx2]),
                marker=None, elinewidth=1, linewidth=0, color='black')
    ax.scatter(time_mid[idx2], (flux_history[idx2] + flux_history_upper_bound[idx2]),
               marker='v', color='black')
    ax.set_xlabel('Date')
    ax.set_ylabel('Flux (ph/cm^2/s)')
    ax.set_ylim(ymin=0)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.figure.autofmt_xdate()

    return ax
