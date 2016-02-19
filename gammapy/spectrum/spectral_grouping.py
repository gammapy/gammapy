# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
from ..spectrum.spectrum_extraction import SpectrumObservationList, SpectrumObservation


__all__ = [
    'SpectralGrouping',
]

log = logging.getLogger(__name__)


class SpectrumGrouping(object):
    """
    """
    def __init__(self, spectrum_observation_list):
        self.spectrum_observation_list = spectrum_observation_list
        self.spectrum_band_list= None
        
    @classmethod
    def define_spectral_groups(self, OffsetRange=[0, 2.5], NOffbin=25, EffRange=[0, 100], NEffbin=40,
                               +                               ZenRange=[0., 70.], NZenbin=30):
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
        CosZenmin = np.cos(Zenmax * math.pi / 180.)
        CosZenmax = np.cos(Zenmin * math.pi / 180.)
        Offtab = Angle(np.linspace(Offmin, Offmax, NOffbin  1), "deg")
        Efftab = Quantity(np.linspace(Effmin, Effmax, NEffbin  1), "")
        CosZentab = Quantity(np.linspace(CosZenmin, CosZenmax, NZenbin  1), "")
        list_obs_group_axis = [ObservationGroupAxis('MUONEFF', Efftab / 100., 'bin_edges'),
                               ObservationGroupAxis('CosZEN', CosZentab, 'bin_edges'),
                               ObservationGroupAxis('Offset', Offtab, 'bin_edges')]
        obs_groups = ObservationGroups(list_obs_group_axis)
        Observation_Table=self.spectrum_observation_list.to_table()
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
                 
                ObsBand=SpectrumObservation.apply_grouping(list_obs_band, nband)
                list_band.append(ObsBand)

        self.spectrum_band_list=SpectrumObservationList(list_band)



        """
    In the SpectrumObservation class
    """
def apply_grouping(self, spectrum_observation_list, ebounds, obs_stacked_id):
 
         """Method that stack the ON, OFF, arf and RMF for an observation group
         
         Parameters
         ----------
         spectrum_observation_list : list of `~gammapy.spectrum.SpectrumObservationList`
            list of the observations to group together
         obs_id: int
             Observation ID for stacked observations
         
         """
         # Stack ON and OFF vector
         on_vec = np.sum([o.on_vector for o in obs_list])
         off_vec = np.sum([o.off_vector for o in obs_list])
         # Stack arf vector
         arf_band = [o.arf_vector * o.livetime for o in obs_list]
         #ATTENTION: DANS LE np.sum IL FAUT FAIRE SELON LA DIMENSION DES OBSERVATIONS CAR IL Y A LA DIMENSION DES ENERGIES ICI
         #arf_band ici est a 2D (erngie*dim(listobservation))
         arf_band_tot = np.sum(arf_band, axis=dimension_observation)
         livetime_tot = np.sum([o.livetime for o in obs_list])
         arf_vec=arf_band_tot/livetime_tot

         # Stack rmf vector
         #Je crois que je dois mettre un .T si je multiplie rmf_mat avec arf car dim rmf_mat is (Etrue,Ereco) et pour multiplier un tableau 2D avec un 1D de dim Etrue, le tableau 2D doit avec la dim (Ereco, Etrue) mais à verifier ladimension de rmf_mat
         rmf_band = [o.rmf_mat.T *o.arf_vector * o.livetime for o in obs_list]
         #ATTENTION: DANS LE np.sum IL FAUT FAIRE SELON LA DIMENSION DES OBSERVATIONS CAR IL Y A LA DIMENSION DES ENERGIES true et des energies reco ICI. Donc rmf_band est a 3D (voir quelle est la dimension des observations)
         rmf_band_tot = np.sum(rmf_band, axis=dimension_observation)
         #ici o.arf_vector*o.livetime est a 2D (dim_E_true*dim_list_observation)
         livetime_arf_tot = np.sum([o.arf_vector*o.livetime for o in obs_list], axis=dimension_observation)
         #Dim de rmf_band_tot est 2D (Etrue,Ereco) ou (Ereco,Etrue). Voir dans quelle sens est la shape pour voir si je peux diviser par un truc de la shape Etrue ou si je dois mettre un .T à rmf_band_tot
         rmf_mat=rmf_band_tot/livetime_arf_tot
         
         # Calculate average alpha
         alpha_band = [o.alpha * o.off_vector.total_counts for o in obs_list]
         alpha_band_tot = np.sum(alpha_band)
         off_tot = np.sum([o.off_vector.total_counts for o in obs_list])
         alpha_mean = alpha_band_tot/off_tot
         off_vec.meta.backscal = 1. / alpha_mean


         #Calculate energy range
         #TODO: pour l instant on va prendre le plus petit range en energy possible pour pas se faire chier avec des livetime different en fonctiond des bins en erngies mais c'est crado. Voir avec Regis aussi c'est quoi cette energie range et si on a vraiment besoin de prendre le max en energy range et de definir un livetime dependant des energies bins
         emin = max([_.meta.energy_range[0] for _ in obs_list])
         emax = min([_.meta.energy_range[1] for _ in obs_list])

         m = Bunch()
         m['energy_range'] = EnergyBounds([emin, emax])
         m['obs_ids'] = [o.obs_id for o in obs_list]
         m['alpha_method1'] = alpha_mean
         m['livetime'] = livetime_tot
         return cls(obs_id, on_vec, off_vec, arf, rmf, meta=m)



     # Loop over the List of SpectrumObservation object to stack the ON, OFF rmf et arf
         ONband = None
         OFFband = None
         OFFtotband = None
         backscalband = None
         livetimeband = None
         arfband = None
         rmfband = None
         for (n, obs) in enumerate(spectrum_observation_list):
             
             # import IPython; IPython.embed()
             on_vector = obs._on.counts
             off_vector = obs._off.counts
             OFF = np.sum(off_vector)
             # For the moment alpha for one band independent of the energy, weighted by the total OFF events
             backscal = obs._off.backscal
             livetime = obs._off.livetime
             arf_vector = obs._aeff.effective_area
             rmf_matrix = obs._edisp.pdf_matrix
             # Find a better way to do this since I initialize for the first SpectrumObservation of the band (n==0) and I sum for the other observations... I think there are a way to combine the initialisation and sum
             if (n == 0):
                 # Pour la creation de l objet effective_area_table et de l objet energy_dispersion pour ecrire en forma ogip
                 energy_hi = obs._aeff.energy_hi
                 energy_lo = obs._aeff.energy_lo
                 # Ca c est tres sale car normalement les membres avec un _ on doit pas y acceder direct comme ca voir comment determiner etrue autrement
                 e_true = obs._edisp._e_true
                 ONband = on_vector
                 OFFband = off_vector
                 OFFtotband = OFF
                 backscalband = backscal * OFF
                 # For a dependent energy backscale
                 # backscalband=backscal*off_vector
                 livetimeband = livetime
                 arfband = arf_vector * livetime
                 # For the first observation to group: the rmftab dimension is initialized to dim(Etrue,Ereco)
                 dim_Etrue = np.shape(rmf_matrix)[0]
                 dim_Ereco = np.shape(rmf_matrix)[1]
                 rmfband = np.zeros((dim_Etrue, dim_Ereco))
                 rmfmean = np.zeros((dim_Etrue, dim_Ereco))
                 for ind_Etrue in range(dim_Etrue):
                     rmfband[ind_Etrue, :] = rmf_matrix[ind_Etrue, :] * arf_vector[ind_Etrue] * livetime
             else:
                 ONband = on_vector
                 OFFband = off_vector
                 OFFtotband = OFF
                 backscalband = backscal * OFF
                 # For a dependent energy backscale
                 # backscalband=backscal*off_vector
                 livetimeband = livetime
                 arfband = arf_vector * livetime
                 # rmf et dimEtrue already defined in the if(n==0) for the first observation
                 for ind_Etrue in range(dim_Etrue):
                     rmfband[ind_Etrue, :] = rmf_matrix[ind_Etrue, :] * arf_vector[ind_Etrue] * livetime
 
         # Mean backscale of the band
         backscalmean = backscalband / OFFtotband
         # backscalmean = backscalband / OFFband
         arfmean = arfband / livetimeband
         for ind_Etrue in range(dim_Etrue):
             rmfmean[ind_Etrue, :] = rmfband[ind_Etrue, :] / arfband[ind_Etrue]
         rmfmean[np.isnan(rmfmean)] = 0
         if (self.obs == 375):
             print rmfmean
             import IPython;
             IPython.embed()
         self._on = CountsSpectrum(ONband, ebounds, livetimeband)
         self._off = CountsSpectrum(OFFband, ebounds, livetimeband, backscalmean)
         self._aeff = EffectiveAreaTable(energy_lo, energy_hi, arfmean)
         self._edisp = EnergyDispersion(rmfmean, e_true, ebounds)        


