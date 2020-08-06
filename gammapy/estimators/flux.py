# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from astropy import units as u
from gammapy.estimators.parameter_estimator import ParameterEstimator
from gammapy.modeling.models import ScaleSpectralModel

log = logging.getLogger(__name__)


class FluxEstimator(ParameterEstimator):
    """Flux estimator.

    Estimates flux for a given list of datasets with their model in a given energy range.

    To estimate the model flux the amplitude of the reference spectral model is
    fitted within the energy range. The amplitude is re-normalized using the "norm" parameter,
    which specifies the deviation of the flux from the reference model in this
    energy range.

    Parameters
    ----------
    source : str or int
        For which source in the model to compute the flux.
    energy_range : `~astropy.units.Quantity`
        the energy interval on which to compute the flux
    norm_min : float
        Minimum value for the norm used for the fit statistic profile evaluation.
    norm_max : float
        Maximum value for the norm used for the fit statistic profile evaluation.
    norm_n_values : int
        Number of norm values used for the fit statistic profile.
    norm_values : `numpy.ndarray`
        Array of norm values to be used for the fit statistic profile.
    n_sigma : int
        Sigma to use for asymmetric error computation.
    n_sigma_ul : int
        Sigma to use for upper limit computation.
    reoptimize : bool
        Re-optimize other free model parameters.
    selection : list of str
        Which additional quantities to estimate. Available options are:

            * "errn-errp": estimate asymmetric errors.
            * "ul": estimate upper limits.
            * "norm-scan": estimate fit statistic profiles.

        By default all steps are executed.
    """

    tag = "FluxEstimator"

    def __init__(
        self,
        source,
        energy_range,
        norm_min=0.2,
        norm_max=5,
        norm_n_values=11,
        norm_values=None,
        n_sigma=1,
        n_sigma_ul=3,
        reoptimize=True,
        selection="all",
    ):

        if norm_values is None:
            norm_values = np.logspace(
                np.log10(norm_min), np.log10(norm_max), norm_n_values
            )
        self.norm_values = norm_values

        self.source = source

        self.energy_range = energy_range

        selection = self._prepare_selection(selection)

        # TODO : check other default parameters
        super().__init__(n_sigma, n_sigma_ul, reoptimize, selection=selection)

    @property
    def energy_range(self):
        return self._energy_range

    @energy_range.setter
    def energy_range(self, energy_range):
        if len(energy_range) != 2:
            raise ValueError("Incorrect size of energy_range")

        emin = u.Quantity(energy_range[0])
        emax = u.Quantity(energy_range[1])

        if emin >= emax:
            raise ValueError("Incorrect energy_range for Flux Estimator")
        self._energy_range = [emin, emax]

    @property
    def e_ref(self):
        return np.sqrt(self.energy_range[0] * self.energy_range[1])

    def __str__(self):
        s = f"{self.__class__.__name__}:\n"
        s += str(self.datasets) + "\n"
        s += str(self.model) + "\n"
        return s

    def _set_model(self, datasets, model):
        # set the model on all datasets
        for dataset in datasets:
            dataset.models[self.source].spectral_model = model

    def _prepare_result(self, model):
        """Prepare the result dictionnary"""
        return {
            "e_ref": self.e_ref,
            "e_min": self.energy_range[0],
            "e_max": self.energy_range[1],
            "ref_dnde": model(self.e_ref),
            "ref_flux": model.integral(self.energy_range[0], self.energy_range[1]),
            "ref_eflux": model.energy_flux(self.energy_range[0], self.energy_range[1]),
            "ref_e2dnde": model(self.e_ref) * self.e_ref ** 2,
        }

    # TODO: do we need norm-scan rather than scan?
    def _prepare_selection(self, selection):
        """Adapt the selection to the ParameterEstimator format."""
        if "norm-scan" in selection:
            selection.remove("norm-scan")
            selection.append("scan")
        return selection

    def run(self, datasets):
        """Estimate flux for a given energy range.

        The fit is performed in the energy range provided by the dataset masks.
        The input energy range is used only to compute the flux normalization.

        Parameters
        ----------
        datasets : list of `~gammapy.spectrum.SpectrumDataset`
            Spectrum datasets.

        Returns
        -------
        result : dict
            Dict with results for the flux point.
        """
        datasets = self._check_datasets(datasets)

        if not datasets.is_all_same_type or not datasets.energy_axes_are_aligned:
            raise ValueError(
                "Flux point estimation requires a list of datasets"
                " of the same type and data shape."
            )
        dataset = datasets[0]

        ref_model = dataset.models[self.source].spectral_model

        scale_model = ScaleSpectralModel(ref_model)
        scale_model.norm.min = 0
        scale_model.norm.max = 1e5

        self._set_model(datasets, scale_model)

        result = self._prepare_result(scale_model.model)

        scale_model.norm.value = 1.0
        scale_model.norm.frozen = False

        result.update(
            super().run(
                datasets, scale_model.norm, null_value=0, scan_values=self.norm_values,
            )
        )
        self._set_model(datasets, ref_model)
        return result

    def _return_nan_result(self, model):
        result = self._prepare_result(model)
        result.update({"norm": np.nan, "stat": np.nan, "success": False})
        result.update({"norm_err": np.nan})
        result.update({"sqrt_ts": np.nan, "ts": np.nan, "null_value": np.nan})
        if "errn-errp" in self.selection:
            result.update({"norm_errp": np.nan, "norm_errn": np.nan})
        if "ul" in self.selection:
            result.update({"norm_ul": np.nan})
        if "scan" in self.selection:
            nans = np.nan * np.empty_like(self.norm_values)
            result.update({"norm_scan": nans, "stat_scan": nans})
        return result
