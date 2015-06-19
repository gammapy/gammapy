# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import astropy.time
import numpy as np


__all__ = ['plot_fermi_3fgl_light_curve',
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


def plot_fermi_3fgl_light_curve(name_3fgl, time_start=None, time_end=None, ax=None):
    """Plot flux as a function of time for a fermi 3FGL object.

    Parameters
    ----------
    name_3FGL : `string`
        The 3FGL catalog name of the object to plot
    time_start : `~astropy.time.Time`
        Light curve start time.  If none, use the earliest time in the catalog.
    time_end : `~astropy.time.Time`
        Light curve end time.  If none, use the latest time in the catalog.
    ax : `~matplotlib.axes.Axes` or None
        Axes

    Returns
    -------
    ax : `~matplotlib.axes.Axes`
        Axes

    Examples
    --------
    Plot effective area vs. energy:

    .. plot::
        :include-source:

        from astropy.time import Time
        from gammapy.time import plot_fermi_3fgl_light_curve
        import matplotlib.pyplot as plt

        time_start = Time('2010-01-01T00:00:00')
        time_end = Time('2015-02-02T02:02:02')

        plt.plot = plot_fermi_3fgl_light_curve('3FGL J0349.9-2102', time_start, time_end)

        plt.show()
    """
    from ..datasets import fetch_fermi_catalog
    from ..time.utils import TIME_REF_FERMI
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    ax = plt.gca() if ax is None else ax

    if time_start is None:
        time_start = astropy.time.Time('2008-08-02T00:33:19')

    if time_end is None:
        time_end = astropy.time.Time('2012-07-31T22:45:47')

    fermi_met_start = (time_start - TIME_REF_FERMI).sec

    fermi_met_end = (time_end - TIME_REF_FERMI).sec

    fermi_cat = fetch_fermi_catalog('3FGL')

    catalog_index = np.where(fermi_cat[1].data['Source_Name'] == name_3fgl)[0][0]

    time_index_start = np.where(fermi_cat[3].data['Hist_Start'] >= fermi_met_start)[0][0]

    # The final entry is the end of the last bin, so no off by one error
    time_index_end = np.where(fermi_cat[3].data['Hist_Start'] <= fermi_met_end)[0][-1] + 1

    time_start = fermi_cat[3].data['Hist_Start'][time_index_start: time_index_end]
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

    # Change bins with no flux value from nan to zero.
    for i in range(0, np.size(flux_history_lower_bound)):
        if np.isnan(flux_history_lower_bound[i]):
            flux_history_lower_bound[i] = 0

    # Create array of upper limits where no lower bound was recorded.
    upper_lims_x = np.copy(time_mid)

    upper_lims_y_start = np.copy(flux_history_upper_bound)

    upper_lims_y = np.copy(flux_history)

    idx = np.where(flux_history_lower_bound <= 0)

    upper_lims_x = upper_lims_x[idx]

    upper_lims_y_start = upper_lims_y_start[idx]

    upper_lims_y = upper_lims_y[idx]

    upper_lims_y_end = np.copy(upper_lims_y_start)

    # Scale the size of the arrow to an arbitrary value.
    upper_lims_y_end *= -0.3

    # Create an array of data points where a lower bound was recorded.
    idx = np.where(flux_history_lower_bound > 0)

    time_mid = time_mid[idx]

    time_diff = time_diff[idx]

    flux_history = flux_history[idx]

    flux_history_upper_bound = flux_history_upper_bound[idx]

    flux_history_lower_bound = flux_history_lower_bound[idx]

    time_mid = (TIME_REF_FERMI + astropy.time.TimeDelta(time_mid, format='sec'))

    time_at_bin_start = time_mid - astropy.time.TimeDelta(time_diff, format='sec')

    time_at_bin_end = time_mid + astropy.time.TimeDelta(time_diff, format='sec')

    time_mid = time_mid.plot_date

    time_at_bin_start = time_at_bin_start.plot_date

    time_at_bin_end = time_at_bin_end.plot_date

    time_diff_at_bin_start = time_mid - time_at_bin_start

    time_diff_at_bin_end = time_at_bin_end - time_mid

    upper_lims_x = (TIME_REF_FERMI
                    + astropy.time.TimeDelta(upper_lims_x, format='sec')).plot_date

    # Plot data points and upper limits.
    plt.errorbar(time_mid, flux_history,
                 yerr=(flux_history_lower_bound, flux_history_upper_bound),
                 xerr=(time_diff_at_bin_start, time_diff_at_bin_end),
                 marker='o', elinewidth=1, linewidth=0, color='black')
    plt.errorbar(upper_lims_x[np.where(upper_lims_y > 0)],
                 upper_lims_y[np.where(upper_lims_y > 0)],
                 yerr=(upper_lims_y_end[np.where(upper_lims_y > 0)],
                       upper_lims_y_start[np.where(upper_lims_y > 0)]),
                 marker='o', elinewidth=1, linewidth=0, lolims=True, color='black')
    plt.errorbar(upper_lims_x[np.where(upper_lims_y <= 0)],
                 upper_lims_y[np.where(upper_lims_y <= 0)],
                 yerr=(upper_lims_y_end[np.where(upper_lims_y <= 0)],
                       upper_lims_y_start[np.where(upper_lims_y <= 0)]),
                 marker=None, elinewidth=1, linewidth=0, lolims=True, color='black')
    plt.xlabel('date')
    plt.ylabel('flux [ph/cm^2/s]')
    plt.gca().xaxis_date()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.gcf().autofmt_xdate()

    return ax
