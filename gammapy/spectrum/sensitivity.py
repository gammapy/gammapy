# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy.table import Column, Table
from gammapy.modeling.models import PowerLawSpectralModel, SkyModel
from gammapy.stats import excess_matching_significance_on_off


__all__ = ["SensitivityEstimator"]


class SensitivityEstimator:
    """Estimate differential sensitivity.

    Uses a 1D spectral analysis and on / off measurement.

    For a usage example see `cta_sensitivity.html <../notebooks/cta_sensitivity.html>`__

    Parameters
    ----------
    alpha : float, optional
        On/OFF normalization
    sigma : float, optional
        Minimum significance
    gamma_min : float, optional
        Minimum number of gamma-rays

    Notes
    -----
    This class allows to determine for each reconstructed energy bin the flux associated to the number of gamma-ray
    events for which the significance is ``sigma``, and being larger than ``gamma_min`` and ``bkg_sys`` percent larger than the
    number of background events in the ON region.
    """

    def __init__(
        self,
        spectrum=None,
        alpha=0.2,
        sigma=5.0,
        gamma_min=10.0,
    ):

        if spectrum is None:
            spectrum = PowerLawSpectralModel()

        self.spectrum = spectrum
        self.alpha = alpha
        self.sigma = sigma
        self.gamma_min = gamma_min

    def estimate_min_excess(self, dataset):
        """Estimate minimum excess to reach the given significance.

        Parameters
        ----------
        dataset : `SpectrumDataset`
            Spectrum dataset

        Return
        ------
        excess : `CountsSpectrum`
            Minimal excess
        """
        n_off = dataset.background.data / self.alpha
        excess_counts = excess_matching_significance_on_off(
            n_off=n_off, alpha=self.alpha, significance=self.sigma
        )
        is_gamma_limited = excess_counts < self.gamma_min
        excess_counts[is_gamma_limited] = self.gamma_min
        return dataset.counts.copy(data=excess_counts)

    def estimate_min_e2dnde(self, excess, dataset):
        """Estimate dnde from given min. excess

        Parameters
        ----------


        """
        energy = dataset.counts.energy.center

        dataset.model = SkyModel(spectral_model=self.spectrum)
        npred = dataset.npred()

        phi_0 = excess / npred

        dnde_model = self.spectrum(energy=energy)
        dnde = (phi_0 * dnde_model * energy ** 2).to("erg / (cm2 s)")
        return dnde

    def _get_criterion(self, excess):
        is_gamma_limited = excess < self.gamma_min
        criterion = np.chararray(excess.shape, itemsize=12)
        criterion[is_gamma_limited] = "gamma"
        criterion[~is_gamma_limited] = "significance"
        return criterion

    def run(self, dataset):
        """Run the sensitivty estimation

        Parameters
        ----------
        dataset : `SpectrumDataset`
            Dataset to compute sensitivty for.

        Returns
        -------
        sensitivity : `~astropy.table.Table`
            Sensitivity table
        """
        energy = dataset.edisp.e_reco.center
        excess = self.estimate_min_excess(dataset)
        e2dnde = self.estimate_min_dnde(excess, dataset)
        criterion = self._get_criterion(excess)

        table = Table(
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
                    data=excess,
                    name="excess",
                    format="5g",
                    description="Number of excess counts in the bin",
                ),
                Column(
                    data=dataset.background.dat,
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
        return table
