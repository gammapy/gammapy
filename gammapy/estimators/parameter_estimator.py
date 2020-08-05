# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from gammapy.datasets import Datasets
from gammapy.modeling import Fit
from .core import Estimator

log = logging.getLogger(__name__)


class ParameterEstimator(Estimator):
    """Model parameter estimator.

    Estimates a model parameter for a group of datasets.
    Compute best fit value, symmetric and asymmetric errors, delta TS for a given null value
    as well as parameter upper limit and fit statistic profile.

    Parameters
    ----------
    n_sigma : int
        Sigma to use for asymmetric error computation. Default is 1.
    n_sigma_ul : int
        Sigma to use for upper limit computation. Default is 2.
    reoptimize : bool
        Re-optimize other free model parameters. Default is True.
    n_scan_values : int
        Number of values used to scan fit stat profile
    scan_n_err : float
        Range to scan in number of parameter error

    """
    tag = "ParameterEstimator"

    def __init__(
        self, n_sigma=1, n_sigma_ul=2, reoptimize=True, n_scan_values=30, scan_n_err=3,
    ):
        self.n_sigma = n_sigma
        self.n_sigma_ul = n_sigma_ul
        self.reoptimize = reoptimize
        self.n_scan_values = n_scan_values
        self.scan_n_err = scan_n_err

    def __str__(self):
        s = f"{self.__class__.__name__}:\n"
        s += str(self.datasets) + "\n"
        return s

    def _check_datasets(self, datasets):
        """Check datasets geometry consistency and return Datasets object"""
        if not isinstance(datasets, Datasets):
            datasets = Datasets(datasets)

        return datasets

    def _freeze_parameters(self, parameter):
        """Freeze all other parameters"""
        for par in self.datasets.parameters:
            if par is not parameter:
                par.frozen = True

    def _compute_scan_values(self, value, value_error, par_min, par_max):
        """Define parameter value range to be scanned"""
        min_range = value - self.scan_n_err * value_error
        if not np.isnan(par_min):
            min_range = np.maximum(par_min, min_range)
        max_range = value + self.scan_n_err * value_error
        if not np.isnan(par_max):
            max_range = np.minimum(par_max, max_range)

        return np.linspace(min_range, max_range, self.n_scan_values)

    def _find_best_fit(self, parameter):
        """Find the best fit solution and store results."""
        fit_result = self.fit.optimize()

        if fit_result.success:
            value = parameter.value
        else:
            value = np.nan

        result = {
            parameter.name: value,
            "stat": fit_result.total_stat,
            "success": fit_result.success,
        }

        self.fit_result = fit_result
        return result

    def _estimate_ts_for_null_value(self, parameter, null_value=1e-150):
        """Returns the fit statistic value for a given null value of the parameter."""
        with self.datasets.parameters.restore_values:
            parameter.value = null_value
            parameter.frozen = True
            result = self.fit.optimize()
        if not result.success:
            log.warning(
                "Fit failed for parameter null value, returning NaN. Check input null value."
            )
            return np.nan
        return result.total_stat

    def run(
        self, datasets, parameter, steps="all", null_value=1e-150, scan_values=None
    ):
        """Run the parameter estimator.

        Parameters
        ----------
        datasets : `~gammapy.datasets.Datasets`
            The datasets used to estimate the model parameter
        parameter : `~gammapy.modeling.Parameter`
            the parameter to be estimated
        steps : list of str
            Which steps to execute. Available options are:
                * "err": estimate symmetric error from covariance
                * "ts": estimate delta TS with parameter null (reference) value
                * "errn-errp": estimate asymmetric errors.
                * "ul": estimate upper limits.
                * "scan": estimate fit statistic profiles.

            By default all steps are executed.
        null_value : float
            the null value to be used for delta TS estimation.
            Default is 1e-150 since 0 can be an issue for some parameters.
        scan_values : `numpy.ndarray`
            Array of parameter values to be used for the fit statistic profile.
            If set to None, scan values are automatically calculated. Default is None.

        Returns
        -------
        result : dict
            Dict with the various parameter estimation values.
        """
        self.datasets = self._check_datasets(datasets)
        self.fit = Fit(datasets)
        self.fit_result = None

        with self.datasets.parameters.restore_values:

            if not self.reoptimize:
                self._freeze_parameters(parameter)

            if steps == "all":
                steps = ["err", "ts", "errn-errp", "ul", "scan"]

            result = self._find_best_fit(parameter)
            TS1 = result["stat"]

            value_max = result[parameter.name]

            if "err" in steps:
                res = self.fit.covariance()
                value_err = res.parameters[parameter].error*self.n_sigma
                result.update({f"{parameter.name}_err": value_err})

            if "errn-errp" in steps:
                res = self.fit.confidence(parameter=parameter, sigma=self.n_sigma)
                result.update(
                    {
                        f"{parameter.name}_errp": res["errp"],
                        f"{parameter.name}_errn": res["errn"],
                    }
                )

            if "ul" in steps:
                res = self.fit.confidence(parameter=parameter, sigma=self.n_sigma_ul)
                result.update({f"{parameter.name}_ul": res["errp"] + value_max})

            if "ts" in steps:
                TS0 = self._estimate_ts_for_null_value(parameter, null_value)
                res = TS0 - TS1
                result.update(
                    {"sqrt_ts": np.sqrt(res), "ts": res, "null_value": null_value}
                )
                # TODO: should not need this
                self.fit.optimize()

            if "scan" in steps:
                if scan_values is None:
                    scan_values = self._compute_scan_values(
                        value_max, value_err, parameter.min, parameter.max
                    )

                res = self.fit.stat_profile(
                    parameter, values=scan_values, reoptimize=self.reoptimize
                )
                result.update(
                    {f"{parameter.name}_scan": res["values"], "stat_scan": res["stat"]}
                )
        return result
