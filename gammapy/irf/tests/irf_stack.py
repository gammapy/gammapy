# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.units import Quantity
from ..irf import EffectiveAreaTable, EnergyDispersion


__all__ = [
    'IRFstack',
]

class IRFstack(object):
    """
    Compute the mean effective area and the mean rmf for a given list of rmf and arf
    """

    def __init__(self, list_arf, list_livetime, list_rmf=None, list_low_threshold=None, list_high_threshold=None):
        self.list_arf=list_arf
        self.list_rmf=list_rmf
        self.list_livetime=list_livetime
        self.stacked_aeff=None
        self.stacked_rmf=None
        self.list_low_threshold=list_low_threshold
        self.list_high_threshold=list_high_threshold

    def mean_arf(self):
        nbins = self.list_arf[0].energy.nbins
        aefft = Quantity(np.zeros(nbins), 'cm2 s')
        for i,arf in enumerate(self.list_arf):
            aeff_data = arf.evaluate(fill_nan=True)
            aefft_current = aeff_data * self.list_livetime[i]
            aefft += aefft_current

        stacked_data = aefft / np.sum(self.list_livetime)
        self.stacked_aeff = EffectiveAreaTable(energy=self.obs_list[0].e_true,
                                               data=stacked_data.to('cm2'))

    def mean_rmf(self):
        reco_bins = self.list_rmf.e_reco.nbins
        true_bins = self.list_rmf.e_true.nbins

        aefft = Quantity(np.zeros(true_bins), 'cm2 s')
        temp = np.zeros(shape=(reco_bins, true_bins))
        aefftedisp = Quantity(temp, 'cm2 s')

        for i,rmf in enumerate(self.list_rmf):
            aeff_data = self.list_arf[i].evaluate(fill_nan=True)
            aefft_current = aeff_data * self.list_livetime[i]
            aefft += aefft_current
            #il faut que je reflechisse plus a ce pb de threshold car pour HAP-FR il y en a pas dans les fits et puis ca doit etre
            #en ereco et donc faut pas prendre le truc par defaut de l'arf
            if not self.list_low_threshold:
            if not self.list_high_threshold:
            if self.list_low_threshold & self.list_high_threshold:
                edisp_data = rmf.pdf_in_safe_range(self.list_low_threshold[i], self.list_high_threshold[i])
            elif not self.list_low_threshold & not self.list_high_threshold:
            elif not self.list_low_threshold:
            elif not self.list_high_threshold:

            aefftedisp += edisp_data.transpose() * aefft_current

        stacked_edisp = np.nan_to_num(aefftedisp / aefft)

        self.stacked_edisp = EnergyDispersion(e_true=self.list_rmf.e_true,
                                              e_reco=elf.list_rmf.e_reco,
                                              data=stacked_edisp.transpose())

