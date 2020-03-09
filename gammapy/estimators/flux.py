# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from astropy import units as u
from gammapy.modeling.models import ScaleSpectralModel
from gammapy.estimators.parameter_estimator import ParameterEstimator

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
    datasets : list of `~gammapy.spectrum.SpectrumDataset`
        Spectrum datasets.
    source : str or int
        For which source in the model to compute the flux points.
    norm_min : float
        Minimum value for the norm used for the fit statistic profile evaluation.
    norm_max : float
        Maximum value for the norm used for the fit statistic profile evaluation.
    norm_n_values : int
        Number of norm values used for the fit statistic profile.
    norm_values : `numpy.ndarray`
        Array of norm values to be used for the fit statistic profile.
    sigma : int
        Sigma to use for asymmetric error computation.
    sigma_ul : int
        Sigma to use for upper limit computation.
    reoptimize : bool
        Re-optimize other free model parameters.
    """

    def __init__(
            self,
            datasets,
            source,
            norm_min=0.2,
            norm_max=5,
            norm_n_values=11,
            norm_values=None,
            sigma=1,
            sigma_ul=3,
            reoptimize=True,
    ):
        # make a copy to not modify the input datasets
        datasets = self._check_datasets(datasets)

        if not (datasets.is_all_same_type and datasets.is_all_same_shape):
            raise ValueError(
                "Flux point estimation requires a list of datasets"
                " of the same type and data shape."
            )

        datasets = datasets.copy()

        dataset = datasets[0]

        model = dataset.models[source].spectral_model

        self.model = ScaleSpectralModel(model)
        self.model.norm.min = 0
        self.model.norm.max = 1e5

        if norm_values is None:
            norm_values = np.logspace(
                np.log10(norm_min), np.log10(norm_max), norm_n_values
            )
        self.norm_values = norm_values

        self.source = source
        datasets = self._set_scale_model(datasets)

        super().__init__(
            datasets,
            sigma,
            sigma_ul,
            reoptimize,
        )

    @property
    def ref_model(self):
        return self.model.model

    def __str__(self):
        s = f"{self.__class__.__name__}:\n"
        s += str(self.datasets) + "\n"
        s += str(self.model) + "\n"
        return s

    def _set_scale_model(self, datasets):
        # set the model on all datasets
        for dataset in datasets:
            dataset.models[self.source].spectral_model = self.model
        return datasets

    def run(self, e_min, e_max, e_ref=None, steps="all"):
        """Estimate flux for a given energy range.

        The fit is performed in the energy range provided by the dataset masks.
        The input energy range is used only to compute the flux normalization.

        Parameters
        ----------
        e_min : `~astropy.units.Quantity`
            the minimum energy of the interval on which to compute the flux
        e_max : `~astropy.units.Quantity`
            the maximum energy of the interval on which to compute the flux
        e_max : `~astropy.units.Quantity`
            the reference energy at which to compute the flux.
            If None, use sqrt(e_min * e_max). Default is None.
        steps : list of str
            Which steps to execute. Available options are:

                * "err": estimate symmetric error.
                * "errn-errp": estimate asymmetric errors.
                * "ul": estimate upper limits.
                * "ts": estimate ts and sqrt(ts) values.
                * "norm-scan": estimate fit statistic profiles.

            By default all steps are executed.

        Returns
        -------
        result : dict
            Dict with results for the flux point.
        """
        e_min = u.Quantity(e_min)
        e_max = u.Quantity(e_max)

        if e_ref is None:
            # Put at log center of the bin
            e_ref = np.sqrt(e_min * e_max)

        result = {
            "e_ref": e_ref,
            "e_min": e_min,
            "e_max": e_max,
            "ref_dnde": self.ref_model(e_ref),
            "ref_flux": self.ref_model.integral(e_min, e_max),
            "ref_eflux": self.ref_model.energy_flux(e_min, e_max),
            "ref_e2dnde": self.ref_model(e_ref) * e_ref ** 2,
        }

        if "norm-scan" in steps:
            steps.remove("norm-scan")
            steps.append("scan")
        if "norm-err" in steps:
            steps.remove("norm-err")
            steps.append("err")
        if steps == "all":
            steps = ["err", "ts", "errp-errn", "ul", "scan"]

        result.update(super().run(self.model.parameters['norm'], steps, null_value=0, scan_values=self.norm_values))
        return result

    def _return_nan_result(self, steps):
        return super().run(self.model.parameters['norm'], steps)
