# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.coordinates import SkyCoord

from .observation import ObservationTable

__all__ = [
    'ObservationTableSummary',
]

class ObservationTableSummary(object):
    """Observation table summary.

    This is an `~astropy.table.Table` sub-class, with a few
    convenience methods. The format of the observation table
    is described in :ref:`dataformats_observation_lists`.

    Parameters:
    -----------
    obs_table:  `~gammapy.data.ObservationTable`
        Observation index table
    target_pos: `~astropy.coordinates.SkyCoord`
        Target position
    """

    def __init__(self, obs_table, target_pos=None):
        self.obs_table = obs_table
        self.target_pos = target_pos

    def plot_zenith_distribution(self, ax=None, bins=100, range=(0,100.)):
        """
        Construct the zenith distribution of the observations

        Returns:
        --------
        ax : `~matplolib.axes`
            Axis
        """
        import matplotlib.pyplot as plt
        ax = plt.gca() if ax is None else ax

        zenith = np.array(self.obs_table['ZEN_PNT'])

        ax.hist(zenith, range=range, bins=bins)
        ax.set_title('Zenith distribution')
        ax.set_xlabel('Zenith (Deg)')
        ax.set_ylabel('#Entries')
        
        return ax

    def plot_offset_distribution(self, ax=None):
        """
        Construct the zenith distribution of the observations

        Returns:
        --------
        ax : `~matplolib.axes`
            Axis
        """
        import matplotlib.pyplot as plt
        ax = plt.gca() if ax is None else ax

        
        
        return ax

    def __str__(self):
        """Summary report"""
        ss = '*** Observation summary ***\n'
        ss += 'Target position: {}\n'.format(self.target_pos)

        # TODO: print some summary stats here ...

        return ss

    def show_in_browser(self):
        """Make HTML file and images in tmp dir, open in browser"""
