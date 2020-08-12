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
    Compute best fit value, symmetric and delta TS for a given null value.
    Additionnally asymmetric errors as well as parameter upper limit and fit statistic profile
    can be estimated.

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
    selection : list of str
        Which additional quantities to estimate. Available options are:
            * "errn-errp": estimate asymmetric errors on parameter best fit value.
            * "ul": estimate upper limits.
            * "scan": estimate fit statistic profiles.

        By default all steps are executed.

    """

    tag = "ParameterEstimator"
    available_selection = ["errn-errp", "ul", "scan"]

    def __init__(
        self,
        parameter,
        n_sigma=1,
        n_sigma_ul=2,
        null_value=1e-150,
        scan_n_sigma=3,
        scan_min=None,
        scan_max=None,
        scan_n_values=30,
        scan_values=None,
        reoptimize=True,
        selection="all",
    ):
        self.parameter = parameter
        self.n_sigma = n_sigma
        self.n_sigma_ul = n_sigma_ul
        self.null_value = null_value

        # scan parameters
        self.scan_n_sigma = scan_n_sigma
        self.scan_n_values = scan_n_values
        self.scan_values = scan_values
        self.scan_min = scan_min
        self.scan_max = scan_max

        self.reoptimize = reoptimize
        self.selection = self._make_selection(selection)
        self._fit = None

    def estimate_best_fit(self, datasets):
        """Estimate parameter assymetric errors

        Parameters
        ----------
        datasets : `~gammapy.datasets.Datasets`
            Datasets

        Returns
        -------
        result : dict
            Dict with the various parameter estimation values.
        """
        # TODO: make Fit stateless
        parameter = datasets.parameters[self.parameter]
        self._fit = Fit(datasets)
        result_fit = self._fit.optimize()
        _ = self._fit.covariance()

        return {
            "value": parameter.value,
            "stat": result_fit.total_stat,
            "success": result_fit.success,
            "err": parameter.error * self.n_sigma,
        }

    def estimate_ts(self, datasets):
        """Estimate parameter ts

        Parameters
        ----------
        datasets : `~gammapy.datasets.Datasets`
            Datasets

        Returns
        -------
        result : dict
            Dict with the various parameter estimation values.
        """
        stat = datasets.stat_sum()

        with datasets.parameters.restore_values:
            parameter = datasets.parameters[self.parameter]

            # compute ts value
            parameter.value = self.null_value

            if self.reoptimize:
                parameter.frozen = True
                _ = self._fit.optimize()

            ts = datasets.stat_sum() - stat

        return {"ts": ts, "sqrt_ts": self.get_sqrt_ts(ts)}

    def estimate_errn_errp(self, datasets):
        """Estimate parameter assymetric errors

        Parameters
        ----------
        datasets : `~gammapy.datasets.Datasets`
            Datasets

        Returns
        -------
        result : dict
            Dict with the various parameter estimation values.
        """
        # TODO: make Fit stateless and configurable
        self._fit.optimize()
        res = self._fit.confidence(
            parameter=self.parameter,
            sigma=self.n_sigma,
            reoptimize=self.reoptimize
        )
        return {
                "errp": res["errp"],
                "errn": res["errn"],
            }

    def estimate_scan(self, datasets):
        """Estimate parameter stat scan.

        Parameters
        ----------
        datasets : `~gammapy.datasets.Datasets`
            The datasets used to estimate the model parameter

        Returns
        -------
        result : dict
            Dict with the various parameter estimation values.

        """
        # TODO: make Fit stateless and configurable
        parameter = datasets.parameters[self.parameter]

        if self.scan_min and self.scan_max:
            bounds = (self.scan_min, self.scan_max)
        else:
            bounds = self.scan_n_sigma

        profile = self._fit.stat_profile(
            parameter=parameter,
            values=self.scan_values,
            bounds=bounds,
            nvalues=self.scan_n_values,
            reoptimize=self.reoptimize
        )

        return {
            "scan": profile["values"],
            "stat_scan": profile["stat"]
        }

    def estimate_ul(self, datasets):
        """Estimate parameter ul.

        Parameters
        ----------
        datasets : `~gammapy.datasets.Datasets`
            The datasets used to estimate the model parameter

        Returns
        -------
        result : dict
            Dict with the various parameter estimation values.

        """
        # TODO: make Fit stateless and configurable
        parameter = datasets.parameters[self.parameter]
        res = self._fit.confidence(parameter=parameter, sigma=self.n_sigma_ul)
        return {"ul": res["errp"] + parameter.value}

    def run(self, datasets):
        """Run the parameter estimator.

        Parameters
        ----------
        datasets : `~gammapy.datasets.Datasets`
            The datasets used to estimate the model parameter

        Returns
        -------
        result : dict
            Dict with the various parameter estimation values.
        """
        datasets = Datasets(datasets)
        parameters = datasets.models.parameters

        with parameters.restore_values:

            if not self.reoptimize:
                parameters.freeze_all()
                parameters[self.parameter].frozen = False

            result = self.estimate_best_fit(datasets)
            result.update(self.estimate_ts(datasets))

            if "errn-errp" in self.selection:
                result.update(self.estimate_errn_errp(datasets))

            if "ul" in self.selection:
                result.update(self.estimate_ul(datasets))

            if "scan" in self.selection:
                result.update(self.estimate_scan(datasets))

        return result
