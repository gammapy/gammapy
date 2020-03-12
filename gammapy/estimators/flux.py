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
        energy_range,
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

        if not datasets.is_all_same_type or not datasets.is_all_same_energy_shape:
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

        self.energy_range = energy_range

        super().__init__(
            datasets, sigma, sigma_ul, reoptimize,
        )
        self._set_scale_model()

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

    @property
    def ref_model(self):
        return self.model.model

    def __str__(self):
        s = f"{self.__class__.__name__}:\n"
        s += str(self.datasets) + "\n"
        s += str(self.model) + "\n"
        return s

    def _set_scale_model(self):
        # set the model on all datasets
        for dataset in self.datasets:
            dataset.models[self.source].spectral_model = self.model

    def _prepare_result(self):
        """Prepare the result dictionnary"""
        return {
            "e_ref": self.e_ref,
            "e_min": self.energy_range[0],
            "e_max": self.energy_range[1],
            "ref_dnde": self.ref_model(self.e_ref),
            "ref_flux": self.ref_model.integral(
                self.energy_range[0], self.energy_range[1]
            ),
            "ref_eflux": self.ref_model.energy_flux(
                self.energy_range[0], self.energy_range[1]
            ),
            "ref_e2dnde": self.ref_model(self.e_ref) * self.e_ref ** 2,
        }

    def _prepare_steps(self, steps):
        """Adapt the steps to the ParameterEstimator format."""
        if "norm-scan" in steps:
            steps.remove("norm-scan")
            steps.append("scan")
        if "norm-err" in steps:
            steps.remove("norm-err")
            steps.append("err")
        if steps == "all":
            steps = ["err", "ts", "errp-errn", "ul", "scan"]
        return steps

    def run(self, steps="all"):
        """Estimate flux for a given energy range.

        The fit is performed in the energy range provided by the dataset masks.
        The input energy range is used only to compute the flux normalization.

        Parameters
        ----------
        steps : list of str
            Which steps to execute. Available options are:

                * "norm-err": estimate symmetric error.
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
        steps = self._prepare_steps(steps)
        result = self._prepare_result()

        self.model.norm.value = 1.05
        self.model.norm.frozen = False

        result.update(
            super().run(
                self.model.norm,
                steps,
                null_value=0,
                scan_values=self.norm_values,
            )
        )
        return result

    def _return_nan_result(self, steps="all"):
        steps = self._prepare_steps(steps)
        result = self._prepare_result()
        result.update({"norm": np.nan, "stat": np.nan, "success": False})
        if "err" in steps:
            result.update({"norm_err": np.nan})
        if "ts" in steps:
            result.update({"sqrt_ts": np.nan, "ts": np.nan, "null_value": np.nan})
        if "errp-errn" in steps:
            result.update({"norm_errp": np.nan, "norm_errn": np.nan})
        if "ul" in steps:
            result.update({"norm_ul": np.nan})
        if "scan" in steps:
            nans = np.nan * np.empty_like(self.norm_values)
            result.update({"norm_scan": nans, "stat_scan": nans})
        return result
