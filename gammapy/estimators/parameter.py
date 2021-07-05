# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
from gammapy.datasets import Datasets
from gammapy.modeling import Fit
from .core import Estimator

log = logging.getLogger(__name__)


class ParameterEstimator(Estimator):
    """Model parameter estimator.

    Estimates a model parameter for a group of datasets. Compute best fit value,
    symmetric and delta TS for a given null value. Additionally asymmetric errors
    as well as parameter upper limit and fit statistic profile can be estimated.

    Parameters
    ----------
    n_sigma : int
        Sigma to use for asymmetric error computation. Default is 1.
    n_sigma_ul : int
        Sigma to use for upper limit computation. Default is 2.
    null_value : float
        Which null value to use for the parameter
    selection_optional : list of str
        Which additional quantities to estimate. Available options are:

            * "all": all the optional steps are executed
            * "errn-errp": estimate asymmetric errors on parameter best fit value.
            * "ul": estimate upper limits.
            * "scan": estimate fit statistic profiles.

        Default is None so the optionnal steps are not executed.
    fit : `Fit`
        Fit instance specifying the backend and fit options.
    reoptimize : bool
        Re-optimize other free model parameters. Default is True.
    """

    tag = "ParameterEstimator"
    _available_selection_optional = ["errn-errp", "ul", "scan"]

    def __init__(
        self,
        n_sigma=1,
        n_sigma_ul=2,
        null_value=1e-150,
        selection_optional=None,
        fit=None,
        reoptimize=True
    ):
        self.n_sigma = n_sigma
        self.n_sigma_ul = n_sigma_ul
        self.null_value = null_value

        self.selection_optional = selection_optional

        if fit is None:
            fit = Fit()

        self.fit = fit
        self.reoptimize = reoptimize

    def estimate_best_fit(self, datasets, parameter):
        """Estimate parameter assymetric errors

        Parameters
        ----------
        datasets : `~gammapy.datasets.Datasets`
            Datasets
        parameter : `Parameter`
            For which parameter to get the value

        Returns
        -------
        result : dict
            Dict with the various parameter estimation values.
        """
        result_fit = self.fit.run(datasets=datasets)

        return {
            f"{parameter.name}": parameter.value,
            "stat": result_fit["optimize_result"].total_stat,
            "success": result_fit["optimize_result"].success,
            f"{parameter.name}_err": parameter.error * self.n_sigma,
        }

    def estimate_ts(self, datasets, parameter):
        """Estimate parameter ts

        Parameters
        ----------
        datasets : `~gammapy.datasets.Datasets`
            Datasets
        parameter : `Parameter`
            For which parameter to get the value

        Returns
        -------
        result : dict
            Dict with the various parameter estimation values.
        """
        stat = datasets.stat_sum()

        with datasets.parameters.restore_status():
            # compute ts value
            parameter.value = self.null_value

            if self.reoptimize:
                parameter.frozen = True
                _ = self.fit.optimize(datasets=datasets)

            ts = datasets.stat_sum() - stat

        return {"ts": ts}

    def estimate_errn_errp(self, datasets, parameter):
        """Estimate parameter assymetric errors

        Parameters
        ----------
        datasets : `~gammapy.datasets.Datasets`
            Datasets
        parameter : `Parameter`
            For which parameter to get the value

        Returns
        -------
        result : dict
            Dict with the various parameter estimation values.
        """
        self.fit.optimize(datasets=datasets)

        res = self.fit.confidence(
            datasets=datasets,
            parameter=parameter,
            sigma=self.n_sigma,
            reoptimize=self.reoptimize
        )
        return {
            f"{parameter.name}_errp": res["errp"],
            f"{parameter.name}_errn": res["errn"],
        }

    def estimate_scan(self, datasets, parameter):
        """Estimate parameter stat scan.

        Parameters
        ----------
        datasets : `~gammapy.datasets.Datasets`
            The datasets used to estimate the model parameter
        parameter : `Parameter`
            For which parameter to get the value

        Returns
        -------
        result : dict
            Dict with the various parameter estimation values.

        """
        self.fit.optimize(datasets=datasets)

        profile = self.fit.stat_profile(
            datasets=datasets,
            parameter=parameter,
            reoptimize=self.reoptimize
        )

        return {
            f"{parameter.name}_scan": profile[f"{parameter.name}_scan"],
            "stat_scan": profile["stat_scan"],
        }

    def estimate_ul(self, datasets, parameter):
        """Estimate parameter ul.

        Parameters
        ----------
        datasets : `~gammapy.datasets.Datasets`
            The datasets used to estimate the model parameter
        parameter : `Parameter`
            For which parameter to get the value

        Returns
        -------
        result : dict
            Dict with the various parameter estimation values.

        """
        self.fit.optimize(datasets=datasets)
        res = self.fit.confidence(
            datasets=datasets,
            parameter=parameter,
            sigma=self.n_sigma_ul,
            reoptimize=self.reoptimize
        )
        return {f"{parameter.name}_ul": res["errp"] + parameter.value}

    def run(self, datasets, parameter):
        """Run the parameter estimator.

        Parameters
        ----------
        datasets : `~gammapy.datasets.Datasets`
            The datasets used to estimate the model parameter
        parameter : `str` or `Parameter`
            For which parameter to run the estimator

        Returns
        -------
        result : dict
            Dict with the various parameter estimation values.
        """
        datasets = Datasets(datasets)
        parameter = datasets.parameters[parameter]

        with datasets.parameters.restore_status():

            if not self.reoptimize:
                datasets.parameters.freeze_all()
                parameter.frozen = False

            result = self.estimate_best_fit(datasets, parameter)
            result.update(self.estimate_ts(datasets, parameter))

            if "errn-errp" in self.selection_optional:
                result.update(self.estimate_errn_errp(datasets, parameter))

            if "ul" in self.selection_optional:
                result.update(self.estimate_ul(datasets, parameter))

            if "scan" in self.selection_optional:
                result.update(self.estimate_scan(datasets, parameter))

        return result
