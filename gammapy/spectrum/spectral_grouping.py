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
    Class that will define the band and the observations in each band from an observation list

    Parameters
    ----------
    spectrum_observation_list : `~gammapy.spectrum.SpectrumObservationList`
        list of SpectrumObservation Object


    """
    def __init__(self, spectrum_observation_list):
        self.spectrum_observation_list = spectrum_observation_list

    def define_spectral_groups(self, OffsetRange=[0, 2.5], NOffbin=25, EffRange=[0, 100], NEffbin=40,
                                                             ZenRange=[0., 70.], NZenbin=30):
        """
    
        Parameters
        ----------
        OffsetRange
        NOffbin
        EffRange
        NEffbin
        NZenbin

        Returns
        -------
        spectrum_band_list: `SpectrumObservationList`
         List of SpectrumObservation for each band
        """
        # Tab contiendrait les bandes et les observations a grouper pour chaque bande
        [Offmin, Offmax] = OffsetRange
        [Effmin, Effmax] = EffRange
        [Zenmin, Zenmax] = ZenRange
        CosZenmin = np.cos(Zenmax * np.pi / 180.)
        CosZenmax = np.cos(Zenmin * np.pi / 180.)
        Offtab = Angle(np.linspace(Offmin, Offmax, NOffbin+1), "deg")
        Efftab = Quantity(np.linspace(Effmin, Effmax, NEffbin+1), "")
        CosZentab = Quantity(np.linspace(CosZenmin, CosZenmax, NZenbin+1), "")
        #list_obs_group_axis = [ObservationGroupAxis('MUONEFF', Efftab / 100., 'bin_edges'),
                               #ObservationGroupAxis('CosZEN', CosZentab, 'bin_edges'),
                               #ObservationGroupAxis('Offset', Offtab, 'bin_edges')]
        list_obs_group_axis = [ObservationGroupAxis('coszen', CosZentab, 'bin_edges'),
                               ObservationGroupAxis('offset', Offtab, 'bin_edges')]
        obs_groups = ObservationGroups(list_obs_group_axis)
        Observation_Table= self.spectrum_observation_list.to_observation_table(True)
        obs_table_grouped = obs_groups.group_observation_table(Observation_Table)
        Nband = obs_groups.n_groups
        list_band = []
        for nband in range(Nband):
            tablegroup = obs_groups.get_group_of_observations(obs_table_grouped, nband)
            if len(tablegroup) == 0:
                continue
            else:
                list_obsid = tablegroup["OBS_ID"]
                list_obs_band = self.spectrum_observation_list.get_obslist_from_obsid(list_obsid)
                 
                ObsBand=SpectrumObservation.grouping_from_an_observation_list(list_obs_band, nband)
                list_band.append(ObsBand)

        return SpectrumObservationList(list_band)


