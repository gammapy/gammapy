# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from gammapy.datasets import Datasets
from gammapy.datasets.actors import DatasetsActor
from gammapy.estimators.parameter import ParameterEstimator
from gammapy.estimators.utils import _get_default_norm
from gammapy.maps import Map, MapAxis
from gammapy.modeling.models import ScaleSpectralModel
from gammapy.utils.deprecation import deprecated_attribute, deprecated_renamed_argument

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
    selection_optional : list of str, optional
        Which additional quantities to estimate. Available options are:

            * "all": all the optional steps are executed.
            * "errn-errp": estimate asymmetric errors.
            * "ul": estimate upper limits.
            * "scan": estimate fit statistic profiles.

        Default is None so the optional steps are not executed.
    fit : `Fit`
        Fit instance specifying the backend and fit options.
    reoptimize : bool
        If True the free parameters of the other models are fitted in each bin independently,
        together with the norm of the source of interest
        (but the other parameters of the source of interest are kept frozen).
        If False only the norm of the source of interest if fitted,
        and all other parameters are frozen at their current values.
        Default is False.
    norm : ~gammapy.modeling.Parameter`
    norm : ~gammapy.modeling.Parameter` or dict
        Norm parameter used for the fit
        Default is None and a new parameter is created automatically,
        with value=1, name="norm", scan_min=0.2, scan_max=5, and scan_n_values = 11.
        By default the min and max are not set and derived from the source model,
        unless the source model does not have one and only one norm parameter.
        If a dict is given the entries should be a subset of
        `~gammapy.modeling.Parameter` arguments.
    """

    tag = "FluxEstimator"

    norm_min = deprecated_attribute("norm_min", "1.2")
    norm_max = deprecated_attribute("norm_max", "1.2")
    norm_n_values = deprecated_attribute("norm_n_values", "1.2")
    norm_values = deprecated_attribute("norm_values", "1.2")

    @deprecated_renamed_argument(
        ["norm_min", "norm_max", "norm_n_values", "norm_values"],
        [None, None, None, None],
        ["1.2", "1.2", "1.2", "1.2"],
    )
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
        norm=None,
    ):

        self.source = source

        self.norm = _get_default_norm(
            norm,
            scan_min=norm_min,
            scan_max=norm_max,
            scan_n_values=norm_n_values,
            scan_values=norm_values,
            interp="log",
        )

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

        if np.isnan(self.norm.min):
            norm.min = scaled_parameter.min / scaled_parameter.value
        if np.isnan(self.norm.max):
            norm.max = scaled_parameter.max / scaled_parameter.value
        norm.interp = scaled_parameter.interp
        return norm

    def get_scale_model(self, models):
        """Set scale model.

        Parameters
        ----------
        models : `Models`
            Models.

        Returns
        -------
        model : `ScaleSpectralModel`
            Scale spectral model.
        """
        ref_model = models[self.source].spectral_model
        scale_model = ScaleSpectralModel(ref_model)

        norms = ref_model.parameters.norm_parameters

        if len(norms.free_parameters) == 1:
            self.norm = self._set_norm_parameter(
                self.norm.copy(), norms.free_parameters[0]
            )

        scale_model.norm = self.norm.copy()
        return scale_model

    def estimate_npred_excess(self, datasets):
        """Estimate npred excess for the source.

        Parameters
        ----------
        datasets : Datasets
            Datasets.

        Returns
        -------
        result : dict
            Dictionary with an array with one entry per dataset with the sum of the
            masked npred excess.
        """
        npred_excess = []

        for dataset in datasets:
            name = datasets.models[self.source].name
            npred_signal = dataset.npred_signal(model_names=[name])
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
            Dictionary with results for the flux point.
        """
        if not isinstance(datasets, DatasetsActor):
            datasets = Datasets(datasets)
        models = datasets.models.copy()

        model = self.get_scale_model(models)

        energy_min, energy_max = datasets.energy_ranges
        energy_axis = MapAxis.from_energy_edges([energy_min.min(), energy_max.max()])

        with np.errstate(invalid="ignore", divide="ignore"):
            result = model.reference_fluxes(energy_axis=energy_axis)
            # convert to scalar values
            result = {key: value.item() for key, value in result.items()}

        # freeze all source model parameters
        models[self.source].parameters.freeze_all()

        models[self.source].spectral_model = model
        datasets.models = models
        result.update(super().run(datasets, model.norm))

        datasets.models[self.source].spectral_model.norm.value = result["norm"]
        result.update(self.estimate_npred_excess(datasets=datasets))
        return result
