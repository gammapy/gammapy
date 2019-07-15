# Licensed under a 3-clause BSD style license - see LICENSE.rst
from astropy.table import Table, Column
import astropy.units as u
from ..stats import excess_matching_significance_on_off
from .models import PowerLaw
from .utils import SpectrumEvaluator

__all__ = ["SensitivityEstimator"]


class SensitivityEstimator:
    """Estimate differential sensitivity.

    Uses a 1D spectral analysis and on / off measurement.

    For a usage example see :gp-notebook:`cta_sensitivity`

    Parameters
    ----------
    arf : `~gammapy.irf.EffectiveAreaTable`
        1D effective area
    rmf : `~gammapy.irf.EnergyDispersion`
        energy dispersion table
    bkg : `~gammapy.spectrum.CountsSpectrum`
        the background array
    livetime : `~astropy.units.Quantity`
        Livetime (object with the units of time), e.g. 5*u.h
    index : float, optional
        Index of the spectral shape (Power-law), should be positive (>0)
    alpha : float, optional
        On/OFF normalisation
    sigma : float, optional
        Minimum significance
    gamma_min : float, optional
        Minimum number of gamma-rays
    bkg_sys : float, optional
        Fraction of Background systematics relative to the number of ON counts

    Notes
    -----
    This class allows to determine for each reconstructed energy bin the flux associated to the number of gamma-ray
    events for which the significance is ``sigma``, and being larger than ``gamma_min`` and ``bkg_sys`` percent larger than the
    number of background events in the ON region.
    """

    def __init__(
        self,
        arf,
        rmf,
        bkg,
        livetime,
        index=2.0,
        alpha=0.2,
        sigma=5.0,
        gamma_min=10.0,
        bkg_sys=0.05,
    ):
        self.arf = arf
        self.rmf = rmf
        self.bkg = bkg
        self.livetime = u.Quantity(livetime).to("s")
        self.index = index
        self.alpha = alpha
        self.sigma = sigma
        self.gamma_min = gamma_min
        self.bkg_sys = bkg_sys

        self._results_table = None

    @property
    def results_table(self):
        """Results table (`~astropy.table.Table`)."""
        return self._results_table

    def run(self):
        """Run the computation."""
        # TODO: let the user decide on energy binning
        # then integrate bkg model and gamma over those energy bins.
        energy = self.rmf.e_reco.center

        bkg_counts = (self.bkg.quantity.to("1/s") * self.livetime).value

        excess_counts = excess_matching_significance_on_off(
            n_off=bkg_counts / self.alpha, alpha=self.alpha, significance=self.sigma
        )
        is_gamma_limited = excess_counts < self.gamma_min
        excess_counts[is_gamma_limited] = self.gamma_min

        model = PowerLaw(
            index=self.index, amplitude="1 cm-2 s-1 TeV-1", reference="1 TeV"
        )

        # TODO: simplify the following computation
        predictor = SpectrumEvaluator(
            model, aeff=self.arf, edisp=self.rmf, livetime=self.livetime
        )
        counts = predictor.compute_npred().data
        phi_0 = excess_counts / counts

        dnde_model = model(energy=energy)
        diff_flux = (phi_0 * dnde_model * energy ** 2).to("erg / (cm2 s)")

        # TODO: take self.bkg_sys into account
        # and add a criterion 'bkg sys'
        criterion = []
        for idx in range(len(energy)):
            if is_gamma_limited[idx]:
                c = "gamma"
            else:
                c = "significance"
            criterion.append(c)

        table = Table(
            [
                Column(
                    data=energy,
                    name="energy",
                    format="5g",
                    description="Reconstructed Energy",
                ),
                Column(
                    data=diff_flux,
                    name="e2dnde",
                    format="5g",
                    description="Energy squared times differential flux",
                ),
                Column(
                    data=excess_counts,
                    name="excess",
                    format="5g",
                    description="Number of excess counts in the bin",
                ),
                Column(
                    data=bkg_counts,
                    name="background",
                    format="5g",
                    description="Number of background counts in the bin",
                ),
                Column(
                    data=criterion,
                    name="criterion",
                    description="Sensitivity-limiting criterion",
                ),
            ]
        )
        self._results_table = table
        return table
