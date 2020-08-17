# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from astropy import units as u
from gammapy.estimators import Estimator
from gammapy.estimators.parameter import ParameterEstimator
from gammapy.modeling.models import ScaleSpectralModel
from gammapy.datasets import Datasets

log = logging.getLogger(__name__)


class FluxEstimator(Estimator):
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
    e_min, e_max: `~astropy.units.Quantity`
        The energy interval on which to compute the flux
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
    selection_optional : list of str
        Which additional quantities to estimate. Available options are:

            * "errn-errp": estimate asymmetric errors.
            * "ul": estimate upper limits.
            * "norm-scan": estimate fit statistic profiles.

        By default all steps are executed.
    """
    tag = "FluxEstimator"
    _available_selection_optional = ["errn-errp", "ul", "scan"]

    def __init__(
        self,
        source,
        e_min,
        e_max,
        norm_min=0.2,
        norm_max=5,
        norm_n_values=11,
        norm_values=None,
        n_sigma=1,
        n_sigma_ul=3,
        reoptimize=True,
        selection_optional="all",
    ):

        if norm_values is None:
            norm_values = np.logspace(
                np.log10(norm_min), np.log10(norm_max), norm_n_values
            )
        self.norm_values = norm_values
        self.source = source
        self.e_min = u.Quantity(e_min)
        self.e_max = u.Quantity(e_max)

        if self.e_min >= self.e_max:
            raise ValueError("Incorrect energy_range for Flux Estimator")

        self.selection_optional = selection_optional
        self.n_sigma = n_sigma
        self.n_sigma_ul = n_sigma_ul
        self.reoptimize = reoptimize

        self._parameter_estimator = ParameterEstimator(
            null_value=0,
            scan_values=self.norm_values,
            n_sigma=self.n_sigma,
            n_sigma_ul=self.n_sigma_ul,
            reoptimize=self.reoptimize,
            selection_optional=self.selection_optional
        )

    @property
    def e_ref(self):
        """Reference energy"""
        return np.sqrt(self.e_min * self.e_max)

    def get_reference_flux_values(self, model):
        """Get reference flux values

        Parameters
        ----------
        model : `SpectralModel`
            Models

        Returns
        -------
        values : dict
            Dictionary with reference energies and flux values.
        """
        return {
            "e_ref": self.e_ref,
            "e_min": self.e_min,
            "e_max": self.e_max,
            "ref_dnde": model(self.e_ref),
            "ref_flux": model.integral(self.e_min, self.e_max),
            "ref_eflux": model.energy_flux(self.e_min, self.e_max),
            "ref_e2dnde": model(self.e_ref) * self.e_ref ** 2,
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
        scale_model.norm.min = 0
        scale_model.norm.max = 1e5
        scale_model.norm.value = 1.0
        scale_model.norm.frozen = False
        return scale_model

    def run(self, datasets):
        """Estimate flux for a given energy range.

        Parameters
        ----------
        datasets : list of `~gammapy.spectrum.SpectrumDataset`
            Spectrum datasets.

        Returns
        -------
        result : dict
            Dict with results for the flux point.
        """
        datasets = Datasets(datasets)

        if not datasets.is_all_same_type or not datasets.energy_axes_are_aligned:
            raise ValueError(
                "Flux point estimation requires a list of datasets"
                " of the same type and data shape."
            )

        model = self.get_scale_model(datasets.models)
        result = self.get_reference_flux_values(model.model)

        for dataset in datasets:
            dataset.models[self.source].spectral_model = model

        result.update(self._parameter_estimator.run(datasets, model.norm))
        return result

    @property
    def nan_result(self):
        result = {
            "norm": np.nan,
            "stat": np.nan,
            "success": False,
            "norm_err": np.nan,
            "ts": np.nan,
        }

        if "errn-errp" in self.selection:
            result.update({"norm_errp": np.nan, "norm_errn": np.nan})

        if "ul" in self.selection:
            result.update({"norm_ul": np.nan})

        if "scan" in self.selection:
            nans = np.nan * np.empty_like(self.norm_values)
            result.update({"norm_scan": nans, "stat_scan": nans})

        return result
