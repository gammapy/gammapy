# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
from astropy.units import Quantity
from ..irf import EffectiveAreaTable, EnergyDispersion

__all__ = [
    'IRFstack',
]

log = logging.getLogger(__name__)


class IRFstack(object):
    """
    Compute the mean effective area and the mean rmf for several observations for a given list of rmf and arf

    Parameters
    ----------
    list_arf: list
        list of arf
    list_livetime: list
        list of livetime
    list_rmf: list
        list of rmf
    list_low_threshold: list
        list of low energy threshold (for the rmf mean computation)
    list_high_threshold: list
        list of high energy threshold (for the rmf mean computation)


    """

    def __init__(self, list_arf, list_livetime, list_rmf=None, list_low_threshold=None, list_high_threshold=None):
        self.list_arf = list_arf
        self.list_rmf = list_rmf
        self.list_livetime = list_livetime
        self.stacked_aeff = None
        self.stacked_edisp = None
        self.list_low_threshold = list_low_threshold
        self.list_high_threshold = list_high_threshold

    def mean_arf(self):
        """
        Compute the mean arf for several observations weighted by the livetime.

        Create a new `~gammapy.irf.EffectiveAreaTable` with the mean arf
        """
        nbins = self.list_arf[0].energy.nbins
        aefft = Quantity(np.zeros(nbins), 'cm2 s')
        livetime_tot = Quantity([t.value for t in self.list_livetime], self.list_livetime[0].unit)
        for i, arf in enumerate(self.list_arf):
            aeff_data = arf.evaluate(fill_nan=True)
            aefft_current = aeff_data * self.list_livetime[i]
            aefft += aefft_current

        stacked_data = aefft / livetime_tot.sum()
        self.stacked_aeff = EffectiveAreaTable(energy=self.list_arf[0].energy,
                                               data=stacked_data.to('cm2'))

    def mean_rmf(self):
        """
        Compute the mean rmf for several observations weighted by the exposure.

        Create a new `~gammapy.irf.EnergyDispersion` with the mean rmf
        """
        reco_bins = self.list_rmf[0].e_reco.nbins
        true_bins = self.list_rmf[0].e_true.nbins

        aefft = Quantity(np.zeros(true_bins), 'cm2 s')
        temp = np.zeros(shape=(reco_bins, true_bins))
        aefftedisp = Quantity(temp, 'cm2 s')

        for i, rmf in enumerate(self.list_rmf):
            aeff_data = self.list_arf[i].evaluate(fill_nan=True)
            aefft_current = aeff_data * self.list_livetime[i]
            aefft += aefft_current
            if (not self.list_low_threshold) & (not self.list_high_threshold):
                log.info('There is no low and high threshold')
            else:
                if not self.list_low_threshold:
                    log.info('There is no low threshold')
                elif not self.list_high_threshold:
                    print('There is no high threshold')
                else:
                    edisp_data = rmf.pdf_in_safe_range(self.list_low_threshold[i], self.list_high_threshold[i])

            aefftedisp += edisp_data.transpose() * aefft_current

        stacked_edisp = np.nan_to_num(aefftedisp / aefft)
        self.stacked_edisp = EnergyDispersion(e_true=self.list_rmf[0].e_true,
                                              e_reco=self.list_rmf[0].e_reco,
                                              data=stacked_edisp.transpose())
