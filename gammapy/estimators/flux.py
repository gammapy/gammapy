# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from gammapy.datasets import Datasets
from gammapy.estimators.parameter import ParameterEstimator
from gammapy.maps import Map, MapAxis
from gammapy.modeling import Parameter, Parameters
from gammapy.modeling.models import ScaleSpectralModel

log = logging.getLogger(__name__)


class FluxEstimator(ParameterEstimator):
    """Flux estimator.

    Estimates flux for a given list of datasets with their model in a given energy range.

    To estimate the model flux the amplitude of the reference spectral model is
    fitted within the energy range. The amplitude is re-normalized using the "norm" parameter,
    which specifies the deviation of the flux from the reference model in this
    energy range.

    Note that there should be only one free norm or amplitude parameter for the estimator to run.

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
        reoptimize=False,
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
            reoptimize=reoptimize,
        )

    def _set_norm_parameter(self, norm=None, scaled_parameter=None):
        """Define properties of the norm spectral parameter."""
        if norm is None:
            norm = Parameter("norm", 1, unit="", interp="log")

        norm.value = 1.0
        norm.frozen = False

        norm.min = scaled_parameter.min / scaled_parameter.value
        norm.max = scaled_parameter.max / scaled_parameter.value
        norm.interp = scaled_parameter.interp
        norm.scan_values = self.norm_values
        norm.scan_min = self.norm_min
        norm.scan_max = self.norm_max
        norm.scan_n_values = self.norm_n_values
        return norm

    def get_scale_model(self, models):
        """Set scale model

        Parameters
        ----------
        models : `Models`
            Models

        Returns
        -------
        model : `ScaleSpectralModel`
            Scale spectral model
        """
        ref_model = models[self.source].spectral_model

        if ref_model.is_norm_spectral_model:
            raise ValueError(
                "Instances of `NormSpectralModel` are not supported for flux point estimation."
            )

        scale_model = ScaleSpectralModel(ref_model)

        norms = Parameters([p for p in ref_model.parameters if p.is_norm])
        if len(norms) == 0 or len(norms.free_parameters) > 1:
            raise ValueError(
                f"{self.tag} requires one and only one free 'norm' or 'amplitude' parameter"
                " in the model to run"
            )
        elif len(norms.free_parameters) == 1:
            norms = norms.free_parameters

        scale_model.norm = self._set_norm_parameter(scale_model.norm, norms[0])
        return scale_model

    def estimate_npred_excess(self, datasets):
        """Estimate npred excess for the source.

        Parameters
        ----------
        datasets : Datasets
            Datasets

        Returns
        -------
        result : dict
            Dict with an array with one entry per dataset with the sum of the
            masked npred excess.
        """
        npred_excess = []

        for dataset in datasets:
            name = datasets.models[self.source].name
            npred_signal = dataset.npred_signal(model_name=name)
            npred = Map.from_geom(dataset.counts.geom)
            npred.stack(npred_signal)
            npred_excess.append(npred.data[dataset.mask].sum())

        return {"npred_excess": np.array(npred_excess), "datasets": datasets.names}

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

        model = self.get_scale_model(models)

        energy_min, energy_max = datasets.energy_ranges
        energy_axis = MapAxis.from_energy_edges([energy_min.min(), energy_max.max()])

        with np.errstate(invalid="ignore", divide="ignore"):
            result = model.reference_fluxes(energy_axis=energy_axis)
            # convert to scalar values
            result = {key: value.item() for key, value in result.items()}

        models[self.source].spectral_model = model
        datasets.models = models
        result.update(super().run(datasets, model.norm))

        datasets.models[self.source].spectral_model.norm.value = result["norm"]
        result.update(self.estimate_npred_excess(datasets=datasets))
        return result
