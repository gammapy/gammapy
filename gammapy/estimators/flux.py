# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from astropy import units as u
from gammapy.datasets import Datasets
from gammapy.estimators import Estimator
from gammapy.estimators.parameter import ParameterEstimator
from gammapy.modeling.models import Models, ScaleSpectralModel

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
    energy_min, energy_max: `~astropy.units.Quantity`
        The energy interval on which to compute the flux
    norm_min : float
        Minimum value for the norm used for the fit.
    norm_max : float
        Maximum value for the norm used for the fit.
    norm_values : `numpy.ndarray`
        Array of norm values to be used for the fit statistic profile.
    n_sigma : int
        Sigma to use for asymmetric error computation.
    n_sigma_ul : int
        Sigma to use for upper limit computation.
    ul_method : {"confidence", "profile"}
        Select upper-limit computation method using confidence or stat profile.
        Default is confidence".
    backend : str
        Backend used for fitting, default : minuit
    optimize_opts : dict
        Options passed to `Fit.optimize`.
    covariance_opts : dict
        Options passed to `Fit.covariance`.
    reoptimize : bool
        Re-optimize other free model parameters.
    selection_optional : list of str
        Which additional quantities to estimate. Available options are:

            * "all": all the optional steps are executed
            * "errn-errp": estimate asymmetric errors.
            * "ul": estimate upper limits.
            * "scan": estimate fit statistic profiles.

        Default is None so the optionnal steps are not executed.
    """

    tag = "FluxEstimator"
    _available_selection_optional = ["errn-errp", "ul", "scan"]

    def __init__(
        self,
        source,
        energy_min,
        energy_max,
        norm_min=1e-8,
        norm_max=1e8,
        norm_values=None,
        n_sigma=1,
        n_sigma_ul=3,
        ul_method="confidence",
        backend="minuit",
        optimize_opts=None,
        covariance_opts=None,
        reoptimize=True,
        selection_optional=None,
    ):

        self.norm_values = norm_values
        self.source = source
        self.energy_min = u.Quantity(energy_min)
        self.energy_max = u.Quantity(energy_max)

        if self.energy_min >= self.energy_max:
            raise ValueError("Incorrect energy_range for Flux Estimator")

        self.n_sigma = n_sigma
        self.n_sigma_ul = n_sigma_ul
        self.backend = backend
        if optimize_opts is None:
            optimize_opts = {}
        if covariance_opts is None:
            covariance_opts = {}
        self.optimize_opts = optimize_opts
        self.covariance_opts = covariance_opts
        self.reoptimize = reoptimize
        self.ul_method = ul_method
        self.selection_optional = selection_optional

    @property
    def _parameter_estimator(self):
        return ParameterEstimator(
            null_value=0,
            scan_values=self.norm_values,
            n_sigma=self.n_sigma,
            n_sigma_ul=self.n_sigma_ul,
            ul_method = self.ul_method,
            backend=self.backend,
            optimize_opts=self.optimize_opts,
            covariance_opts=self.covariance_opts,
            reoptimize=self.reoptimize,
            selection_optional=self.selection_optional,
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
        scale_model.norm.min = self.norm_min
        scale_model.norm.max = self.norm_max
        scale_model.norm.value = 1.0
        scale_model.norm.frozen = False
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

        datasets_sliced = datasets.slice_by_energy(
            energy_min=self.energy_min, energy_max=self.energy_max
        )

        models = datasets.models.copy()
        datasets_sliced.models = models

        if len(datasets_sliced) > 0:
            # TODO: this relies on the energy binning of the first dataset
            energy_axis = datasets_sliced[0].counts.geom.axes["energy"]
            energy_min, energy_max = energy_axis.edges.min(), energy_axis.edges.max()
        else:
            energy_min, energy_max = self.energy_min, self.energy_max

        contributions = []

        for dataset in datasets_sliced:
            if dataset.mask is not None:
                value = dataset.mask.data.any()
            else:
                value = True
            contributions.append(value)

        model = self.get_scale_model(models)

        with np.errstate(invalid="ignore", divide="ignore"):
            result = self.get_reference_flux_values(model.model, energy_min, energy_max)

        if len(datasets) == 0 or not np.any(contributions):
            result.update(self.nan_result)
        else:
            models[self.source].spectral_model = model

            datasets_sliced.models = models
            result.update(self._parameter_estimator.run(datasets_sliced, model.norm))
            result["sqrt_ts"] = self.get_sqrt_ts(result["ts"], result["norm"])

        return result

    @property
    def nan_result(self):
        result = {
            "norm": np.nan,
            "stat": np.nan,
            "success": False,
            "norm_err": np.nan,
            "ts": np.nan,
            "sqrt_ts": np.nan,
        }

        if "errn-errp" in self.selection_optional:
            result.update({"norm_errp": np.nan, "norm_errn": np.nan})

        if "ul" in self.selection_optional:
            result.update({"norm_ul": np.nan})

        if "scan" in self.selection_optional:
            nans = np.nan * np.empty_like(self.norm_values)
            result.update({"norm_scan": nans, "stat_scan": nans})

        return result
