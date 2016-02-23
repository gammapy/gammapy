# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from astropy.coordinates import Angle
from astropy.units import Quantity
import numpy as np
from ..spectrum.spectrum_extraction import SpectrumObservationList, SpectrumObservation
from ..data import ObservationGroupAxis, ObservationGroups

__all__ = [
    'SpectralGrouping',
]

log = logging.getLogger(__name__)


class SpectralGrouping(object):
    """
    Class that will define the band and the observations in each band from an observation list.

    Parameters
    ----------
    spectrum_observation_list : `~gammapy.spectrum.SpectrumObservationList`
        list of SpectrumObservation Object

    """

    def __init__(self, spectrum_observation_list):
        self.spectrum_observation_list = spectrum_observation_list

    def define_spectral_groups(self, offset_range=[0, 2.5], n_off_bin=25, eff_range=[0, 100], n_eff_bin=40,
                               zen_range=[0., 70.], n_zen_bin=30):
        """Define the number of bands in zenith, efficiency and offset.
    
        Parameters
        ----------
        offset_range : tuple
            Min/Max Offset boundaries for the band
        n_off_bin : int
            Number of offset bin
        eff_range : tuple
            Min/Max Efficiency boundaries for the band
        n_eff_bin : int
            Number of efficiency bin
        zen_range : tuple
            Min/Max zenith angle boundaries for the band
        n_zen_bin : int
            Number of zenith bin

        Returns
        -------
        obs_group: `~gammapy.data.ObservationGroups`

        """
        [offmin, offmax] = offset_range
        [effmin, effmax] = eff_range
        [zenmin, zenmax] = zen_range
        coszenmin = np.cos(zenmax * np.pi / 180.)
        coszenmax = np.cos(zenmin * np.pi / 180.)
        offtab = Angle(np.linspace(offmin, offmax, n_off_bin + 1), "deg")
        efftab = Quantity(np.linspace(effmin, effmax, n_eff_bin + 1), "")
        coszentab = Quantity(np.linspace(coszenmin, coszenmax, n_zen_bin + 1), "")
        list_obs_group_axis = [ObservationGroupAxis('muoneff', efftab / 100., 'bin_edges'),
                               ObservationGroupAxis('coszen', coszentab, 'bin_edges'),
                               ObservationGroupAxis('offset', offtab, 'bin_edges')]
        # list_obs_group_axis = [ObservationGroupAxis('coszen', CosZentab, 'bin_edges'),
        # ObservationGroupAxis('offset', Offtab, 'bin_edges')]
        obs_groups = ObservationGroups(list_obs_group_axis)
        return obs_groups

    def apply_grouping(self, obs_groups):
        """Attribute the number of the band to each observation and stack the observations together.

        Parameters
        ----------
        obs_groups: `~gammapy.data.ObservationGroups`
                Contains the boundaries of the different band

        Returns
        -------
        list_band : `~gammapy.spectrum.SpectrumObservationList`
            List of SpectrumObservation for each band

        """
        # Define a new observation table with the number of each band for each observation
        observation_table = self.spectrum_observation_list.to_observation_table(True)
        obs_table_grouped = obs_groups.group_observation_table(observation_table)

        nband = obs_groups.n_groups
        list_band = []
        for nb in range(nband):
            tablegroup = obs_groups.get_group_of_observations(obs_table_grouped, nb)
            # If no observation in the band, pass
            if len(tablegroup) == 0:
                continue
            else:
                list_obsid = tablegroup["OBS_ID"]
                list_obs_band = self.spectrum_observation_list.get_obslist_from_obsid(list_obsid)

                obsband = SpectrumObservation.grouping_from_an_observation_list(list_obs_band, nb)
                list_band.append(obsband)

        return SpectrumObservationList(list_band)

    def define_groups_and_stack(self, offset_range=[0, 2.5], n_off_bin=25, eff_range=[0, 100], n_eff_bin=40,
                                zen_range=[0., 70.], n_zen_bin=30):
        """Define the number of bands in zenith, efficiency and offset and stack the events in each band
        using the previous method

        Parameters
        ----------
        offset_range : tuple
            Min/Max Offset boundaries for the band
        n_off_bin : int
            Number of offset bin
        eff_range : tuple
            Min/Max Efficiency boundaries for the band
        n_eff_bin : int
            Number of efficiency bin
        zen_range : tuple
            Min/Max zenith angle boundaries for the band
        n_zen_bin : int
            Number of zenith bin

        Returns
        -------
        list_band : `~gammapy.spectrum.SpectrumObservationList`
            List of SpectrumObservation for each band

        """
        obs_group = self.define_spectral_groups(offset_range, n_off_bin, eff_range, n_eff_bin, zen_range, n_zen_bin)
        list_obs_band = self.apply_grouping(obs_group)
        return list_obs_band
