# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


__all__ = ['plot_light_curve',
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

def plot_light_curve(name_3FGL, time_start, time_end):
    """Plot flux as a function of time for a fermi 3FGL object.

    Parameters
    ----------
    name_3FGL : `string`
        The 3FGL catalog name of the object to plot
    fermi_met_start : `~astropy.time.Time`
        Astropy time object for the start of the light curve
    fermi_met_end : `~astropy.time.Time`
        Astropy time object for the end of the light curve

    Usage
    gammapy.time.plot_light_curve('3FGL J0349.9-2102', 2.1e8, 3.2e8)
    """
    from ..datasets import fetch_fermi_catalog
    import astropy.time
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import math

    fermi_met_base = astropy.time.Time('2001-01-01T00:00:00')

    fermi_met_start = (time_start - fermi_met_base).sec

    fermi_met_end = (time_end - fermi_met_base).sec

    # As far as I know light curves were only included in 3FGL, so we must use this catalog.
    fermi_cat = fetch_fermi_catalog('3FGL')

    catalog_index = np.where(fermi_cat[1].data['Source_Name'] == name_3FGL)[0][0]

    time_index_start = np.where(fermi_cat[3].data['Hist_Start'] >= fermi_met_start)[0][0]

    # The final entry is the end of the last bin, so no off by one error
    time_index_end = np.where(fermi_cat[3].data['Hist_Start'] <= fermi_met_end)[0][-1] + 1

    time_start = fermi_cat[3].data['Hist_Start'][time_index_start: time_index_end]
    time_end = np.roll(time_start, -1)

    time_diff = time_end - time_start

    # Trim because there is one more bin edge than there is bin mid point
    time_diff = time_diff[0:-1]

    # Midpoints of each bin.
    time_mid = time_start[0:-1] + time_diff

    flux_history = fermi_cat[1].data[catalog_index]['Flux_History'][time_index_start: time_index_end]

    flux_history_lower_bound = fermi_cat[1].data[catalog_index]['Unc_Flux_History'][time_index_start: time_index_end, 0]
    flux_history_upper_bound = fermi_cat[1].data[catalog_index]['Unc_Flux_History'][time_index_start: time_index_end, 1]
    flux_history_lower_bound = abs(flux_history_lower_bound)

    # Change bins with no flux value from nan to zero.
    for i in range(0, np.size(flux_history_lower_bound)):
        if math.isnan(flux_history_lower_bound[i]):
            flux_history_lower_bound[i] = 0

    # Create array of upper limits where no lower bound was recorded.
    upper_lims_x = np.copy(time_mid)

    upper_lims_y_start = np.copy(flux_history_upper_bound)

    upper_lims_y = np.copy(flux_history)

    upper_lims_x = \
        upper_lims_x[np.where(flux_history_lower_bound <= 0)]

    upper_lims_y_start = \
        upper_lims_y_start[np.where(flux_history_lower_bound <= 0)]

    upper_lims_y = \
        upper_lims_y[np.where(flux_history_lower_bound <= 0)]

    upper_lims_y_end = np.copy(upper_lims_y_start)

    # Scale the size of the arrow to an arbitrary value.
    upper_lims_y_end *= -0.3

    # Create an array of data points where a lower bound was recorded.
    time_mid = \
        time_mid[np.where((flux_history_lower_bound) > 0)]

    time_diff = \
        time_diff[np.where((flux_history_lower_bound) > 0)]

    flux_history = \
        flux_history[np.where((flux_history_lower_bound) > 0)]

    flux_history_upper_bound = \
        flux_history_upper_bound[np.where((flux_history_lower_bound) > 0)]

    flux_history_lower_bound = \
        flux_history_lower_bound[np.where((flux_history_lower_bound) > 0)]

    time_mid = (fermi_met_base + astropy.time.TimeDelta(time_mid, format='sec')).plot_date
    upper_lims_x = (fermi_met_base + astropy.time.TimeDelta(upper_lims_x, format='sec')).plot_date


    # Plot data points and upper limits.
    plt.errorbar(time_mid, flux_history, yerr=(flux_history_lower_bound, flux_history_upper_bound), \
                 #xerr=0.5 * time_diff, \
                 marker='o', elinewidth=1, linewidth=0, color='black')
    plt.errorbar(upper_lims_x[np.where(upper_lims_y > 0)], upper_lims_y[np.where(upper_lims_y > 0)], \
                 yerr=(upper_lims_y_end[np.where(upper_lims_y > 0)], upper_lims_y_start[np.where(upper_lims_y > 0)]), \
                 marker='o', elinewidth=1, linewidth=0, lolims=True, color='black')
    plt.errorbar(upper_lims_x[np.where(upper_lims_y <= 0)], upper_lims_y[np.where(upper_lims_y <= 0)], \
                 yerr=(upper_lims_y_end[np.where(upper_lims_y <= 0)], upper_lims_y_start[np.where(upper_lims_y <= 0)]), \
                 marker=None, elinewidth=1, linewidth=0, lolims=True, color='black')
    plt.xlabel('date')
    plt.ylabel('flux [ph/cm^2/s]')
    plt.gca().xaxis_date()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.gcf().autofmt_xdate()
    plt.show()
