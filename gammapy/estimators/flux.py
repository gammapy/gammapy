# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from astropy import units as u
from gammapy.datasets import Datasets
from gammapy.estimators import Estimator
from gammapy.estimators.parameter import ParameterEstimator
from gammapy.modeling.models import Models, ScaleSpectralModel
from gammapy.modeling import Fit

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
    selection_optional : list of str
        Which additional quantities to estimate. Available options are:

            * "all": all the optional steps are executed
            * "errn-errp": estimate asymmetric errors.
            * "ul": estimate upper limits.
            * "scan": estimate fit statistic profiles.

        Default is None so the optional steps are not executed.
    fit : `Fit`
        Fit instance specifying the backend and fit options.
    reoptimize : bool
        Re-optimize other free model parameters. Default is False.
    """
    tag = "FluxEstimator"
    _available_selection_optional = ["errn-errp", "ul", "scan"]

    def __init__(
        self,
        source=0,
        norm_min=0.2,
        norm_max=5,
        norm_n_values=11,
        norm_values=None,
        n_sigma=1,
        n_sigma_ul=2,
        selection_optional=None,
        fit=None,
        reoptimize=False
    ):
        self.norm_values = norm_values
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.norm_n_values = norm_n_values
        self.source = source
        super().__init__(
            null_value=0,
            n_sigma=n_sigma,
            n_sigma_ul=n_sigma_ul,
            selection_optional=selection_optional,
            fit=fit,
            reoptimize=reoptimize
        )

    @staticmethod
    def get_reference_flux_values(model, energy_min, energy_max):
        """Get reference flux values

        Parameters
        ----------
        model : `SpectralModel`
            Models
        energy_min, energy_max : `~astropy.units.Quantity`
            Energy range

        Returns
        -------
        values : dict
            Dictionary with reference energies and flux values.
        """
        energy_ref = np.sqrt(energy_min * energy_max)
        return {
            "e_ref": energy_ref,
            "e_min": energy_min,
            "e_max": energy_max,
            "ref_dnde": model(energy_ref),
            "ref_flux": model.integral(energy_min, energy_max),
            "ref_eflux": model.energy_flux(energy_min, energy_max),
            "ref_e2dnde": model(energy_ref) * energy_ref ** 2,
        }

    def get_scale_model(self, models):
        """Set scale model

        Parameters
        ----------
        models : `Models`
            Models

        Return
        ------
        model : `ScaleSpectralModel`
            Scale spectral model
        """
        ref_model = models[self.source].spectral_model
        scale_model = ScaleSpectralModel(ref_model)
        scale_model.norm.value = 1.0
        scale_model.norm.frozen = False
        scale_model.norm.scan_values = self.norm_values
        scale_model.norm.interp = "log"
        scale_model.norm.scan_min = self.norm_min
        scale_model.norm.scan_max = self.norm_max
        scale_model.norm.scan_n_values = self.norm_n_values
        return scale_model

    def run(self, datasets):
        """Estimate flux for a given energy range.

        Parameters
        ----------
        datasets : list of `~gammapy.datasets.SpectrumDataset`
            Spectrum datasets.

        Returns
        -------
        result : dict
            Dict with results for the flux point.
        """
        datasets = Datasets(datasets)
        models = datasets.models.copy()

        contributions = []

        for dataset in datasets:
            if dataset.mask is not None:
                value = dataset.mask.data.any()
            else:
                value = True
            contributions.append(value)

        model = self.get_scale_model(models)

        energy_min, energy_max = datasets.energy_ranges

        with np.errstate(invalid="ignore", divide="ignore"):
            result = self.get_reference_flux_values(
                model.model, energy_min.min(), energy_max.min()
            )

        if len(datasets) == 0 or not np.any(contributions):
            result.update(self.nan_result(datasets, model.norm))
        else:
            models[self.source].spectral_model = model

            datasets.models = models
            result.update(super().run(datasets, model.norm))

        return result

    def nan_result(self, datasets, norm):
        """Nan result"""
        result = {
            "norm": np.nan,
            "stat": np.nan,
            "success": False,
            "norm_err": np.nan,
            "ts": np.nan,
            "counts": np.zeros(len(datasets))
        }

        if "errn-errp" in self.selection_optional:
            result.update({"norm_errp": np.nan, "norm_errn": np.nan})

        if "ul" in self.selection_optional:
            result.update({"norm_ul": np.nan})

        if "scan" in self.selection_optional:
            norm_scan = norm.scan_values
            result.update({"norm_scan": norm_scan, "stat_scan": np.nan * norm_scan})

        return result
