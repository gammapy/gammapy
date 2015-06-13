# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from ..datasets import fetch_fermi_catalog

__all__ = ['plot_light_curve',
           ]


def plot_light_curve(name_3FGL, fermi_met_start = 2.3933e8, fermi_met_end = 3.65467550e8):
    """Plot flux as a function of time for a fermi 3FGL object.

    Parameters
    ----------
    name_3FGL : `string`
        The 3FGL catalog name of the object to plot
    fermi_met_start : `int`
        Start time of the light curve in Fermi MET
    fermi_met_end : `int`
        End time of the light curve in Fermi MET

    Usage
    gammapy.time.plot_light_curve('3FGL J0349.9-2102', 2.1e8, 3.2e8)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import math

    try:
        fermi_met_start = int(fermi_met_start)
    except:
        print("Fermi_met_start not a valid numerical input.")
        return

    try:
        fermi_met_end = int(fermi_met_end)
    except:
        print("Fermi_met_end not a valid numerical input.")
        return

    # As far as I know light curves were only included in 3FGL, so we must use this catalog.
    fermi_cat = fetch_fermi_catalog('3FGL')

    try:
        catalog_index = np.where(fermi_cat[1].data['Source_Name'] == name_3FGL)[0][0]
    except IndexError:
        print("Could not find " + name_3FGL + " in the 3GL catalog.")
        return

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

    # Plot data points and upper limits.
    plt.errorbar(time_mid, flux_history, yerr=(flux_history_lower_bound, flux_history_upper_bound), \
                 xerr=0.5 * time_diff, \
                 marker='o', elinewidth=1, linewidth=0, color='black')
    plt.errorbar(upper_lims_x[np.where(upper_lims_y > 0)], upper_lims_y[np.where(upper_lims_y > 0)], \
                 yerr=(upper_lims_y_end[np.where(upper_lims_y > 0)], upper_lims_y_start[np.where(upper_lims_y > 0)]), \
                 marker='o', elinewidth=1, linewidth=0, lolims=True, color='black')
    plt.errorbar(upper_lims_x[np.where(upper_lims_y <= 0)], upper_lims_y[np.where(upper_lims_y <= 0)], \
                 yerr=(upper_lims_y_end[np.where(upper_lims_y <= 0)], upper_lims_y_start[np.where(upper_lims_y <= 0)]), \
                 marker=None, elinewidth=1, linewidth=0, lolims=True, color='black')
    plt.xlabel('time [Fermi MET]')
    plt.ylabel('flux [ph/cm^2/s]')
    plt.show()
