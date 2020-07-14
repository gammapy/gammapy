# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy.table import Column, Table
from gammapy.modeling.models import PowerLawSpectralModel, SkyModel
from gammapy.stats import WStatCountsStatistic
from .core import Estimator


__all__ = ["SensitivityEstimator"]


class SensitivityEstimator(Estimator):
    """Estimate differential sensitivity.

    This class allows to determine for each reconstructed energy bin the flux
    associated to the number of gamma-ray events for which the significance is
    ``sigma``, and being larger than ``gamma_min`` and ``bkg_sys`` percent
    larger than the number of background events in the ON region.


    Parameters
    ----------
    spectrum : `SpectralModel`
        Spectral model assumption
    sigma : float, optional
        Minimum significance
    gamma_min : float, optional
        Minimum number of gamma-rays

    Examples
    --------
    For a usage example see `cta_sensitivity.html <../tutorials/cta_sensitivity.html>`__

    """
    tag = "SensitivityEstimator"

    def __init__(
        self, spectrum=None, sigma=5.0, gamma_min=10,
    ):

        if spectrum is None:
            spectrum = PowerLawSpectralModel(index=2, amplitude="1 cm-2 s-1 TeV-1")

        self.spectrum = spectrum
        self.sigma = sigma
        self.gamma_min = gamma_min

    def estimate_min_excess(self, dataset):
        """Estimate minimum excess to reach the given significance.

        Parameters
        ----------
        dataset : `SpectrumDataset`
            Spectrum dataset

        Returns
        -------
        excess : `RegionNDMap`
            Minimal excess
        """
        n_off = dataset.counts_off.data

        stat = WStatCountsStatistic(
            n_on=np.ones_like(n_off), n_off=n_off, alpha=dataset.alpha.data
        )
        excess_counts = stat.excess_matching_significance(self.sigma)
        is_gamma_limited = excess_counts < self.gamma_min
        excess_counts[is_gamma_limited] = self.gamma_min
        excess = dataset.background.copy()
        excess.data = excess_counts
        return excess

    def estimate_min_e2dnde(self, excess, dataset):
        """Estimate dnde from given min. excess

        Parameters
        ----------
        excess : `RegionNDMap`
            Minimal excess
        dataset : `SpectrumDataset`
            Spectrum dataset

        Returns
        -------
        e2dnde : `~astropy.units.Quantity`
            Minimal differential flux.
        """
        energy = dataset.background.geom.axes[0].center

        dataset.models = SkyModel(spectral_model=self.spectrum)
        npred = dataset.npred()

        phi_0 = excess / npred

        dnde_model = self.spectrum(energy=energy)
        dnde = phi_0.data[:, 0, 0] * dnde_model * energy ** 2
        return dnde.to("erg / (cm2 s)")

    def _get_criterion(self, excess):
        is_gamma_limited = excess == self.gamma_min
        criterion = np.chararray(excess.shape, itemsize=12)
        criterion[is_gamma_limited] = "gamma"
        criterion[~is_gamma_limited] = "significance"
        return criterion

    def run(self, dataset):
        """Run the sensitivty estimation

        Parameters
        ----------
        dataset : `SpectrumDatasetOnOff`
            Dataset to compute sensitivty for.

        Returns
        -------
        sensitivity : `~astropy.table.Table`
            Sensitivity table
        """
        energy = dataset._edisp_kernel.e_reco.center
        excess = self.estimate_min_excess(dataset)
        e2dnde = self.estimate_min_e2dnde(excess, dataset)
        criterion = self._get_criterion(excess.data.squeeze())

        return Table(
            [
                Column(
                    data=energy,
                    name="energy",
                    format="5g",
                    description="Reconstructed Energy",
                ),
                Column(
                    data=e2dnde,
                    name="e2dnde",
                    format="5g",
                    description="Energy squared times differential flux",
                ),
                Column(
                    data=excess.data.squeeze(),
                    name="excess",
                    format="5g",
                    description="Number of excess counts in the bin",
                ),
                Column(
                    data=dataset.background.data.squeeze(),
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
