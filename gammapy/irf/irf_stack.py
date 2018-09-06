# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
from astropy.units import Quantity
from ..irf import EffectiveAreaTable, EnergyDispersion

__all__ = ["IRFStacker"]

log = logging.getLogger(__name__)


class IRFStacker(object):
    r"""
    Stack instrument response functions.

    Compute mean effective area and the mean energy dispersion for a given for a
    given list of instrument response functions. Results are stored as
    attributes.

    The stacking of :math:`j` elements is implemented as follows.  :math:`k`
    and :math:`l` denote a bin in reconstructed and true energy, respectively.

    .. math::

        \epsilon_{jk} =\left\{\begin{array}{cl} 1, & \mbox{if
            bin k is inside the energy thresholds}\\ 0, & \mbox{otherwise} \end{array}\right.

        \overline{t} = \sum_{j} t_i

        \overline{\mathrm{aeff}}_l = \frac{\sum_{j}\mathrm{aeff}_{jl}
            \cdot t_j}{\overline{t}}

        \overline{\mathrm{edisp}}_{kl} = \frac{\sum_{j} \mathrm{edisp}_{jkl}
            \cdot \mathrm{aeff}_{jl} \cdot t_j \cdot \epsilon_{jk}}{\sum_{j} \mathrm{aeff}_{jl}
            \cdot t_j}

    Parameters
    ----------
    list_aeff : list
        list of `~gammapy.irf.EffectiveAreaTable`
    list_livetime : list
        list of `~astropy.units.Quantity` (livetime)
    list_edisp : list
        list of `~gammapy.irf.EnergyDispersion`
    list_low_threshold : list
        list of low energy threshold, optional for effective area mean computation
    list_high_threshold : list
        list of high energy threshold, optional for effective area mean computation
    """

    def __init__(
        self,
        list_aeff,
        list_livetime,
        list_edisp=None,
        list_low_threshold=None,
        list_high_threshold=None,
    ):
        self.list_aeff = list_aeff
        self.list_livetime = Quantity(list_livetime)
        self.list_edisp = list_edisp
        self.list_low_threshold = list_low_threshold
        self.list_high_threshold = list_high_threshold
        self.stacked_aeff = None
        self.stacked_edisp = None

    def stack_aeff(self):
        """
        Compute mean effective area (`~gammapy.irf.EffectiveAreaTable`).
        """
        nbins = self.list_aeff[0].energy.nbins
        aefft = Quantity(np.zeros(nbins), "cm2 s")
        livetime_tot = np.sum(self.list_livetime)

        for i, aeff in enumerate(self.list_aeff):
            aeff_data = aeff.evaluate_fill_nan()
            aefft_current = aeff_data * self.list_livetime[i]
            aefft += aefft_current

        stacked_data = aefft / livetime_tot
        self.stacked_aeff = EffectiveAreaTable(
            energy_lo=self.list_aeff[0].energy.lo,
            energy_hi=self.list_aeff[0].energy.hi,
            data=stacked_data.to("cm2"),
        )

    def stack_edisp(self):
        """
        Compute mean energy dispersion (`~gammapy.irf.EnergyDispersion`).
        """
        reco_bins = self.list_edisp[0].e_reco.nbins
        true_bins = self.list_edisp[0].e_true.nbins

        aefft = Quantity(np.zeros(true_bins), "cm2 s")
        temp = np.zeros(shape=(reco_bins, true_bins))
        aefftedisp = Quantity(temp, "cm2 s")

        for i, edisp in enumerate(self.list_edisp):
            aeff_data = self.list_aeff[i].evaluate_fill_nan()
            aefft_current = aeff_data * self.list_livetime[i]
            aefft += aefft_current
            edisp_data = edisp.pdf_in_safe_range(
                self.list_low_threshold[i], self.list_high_threshold[i]
            )

            aefftedisp += edisp_data.transpose() * aefft_current

        with np.errstate(divide="ignore", invalid="ignore"):
            stacked_edisp = np.nan_to_num(aefftedisp / aefft)

        self.stacked_edisp = EnergyDispersion(
            e_true_lo=self.list_edisp[0].e_true.lo,
            e_true_hi=self.list_edisp[0].e_true.hi,
            e_reco_lo=self.list_edisp[0].e_reco.lo,
            e_reco_hi=self.list_edisp[0].e_reco.hi,
            data=stacked_edisp.transpose(),
        )
