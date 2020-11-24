# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
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
    null_value : float
        Which null value to use for the parameter
    scan_n_sigma : int
        Range to scan in number of parameter error
    scan_min : float
        Minimum value to use for the stat scan
    scan_max : int
        Maximum value to use for the stat scan
    scan_n_values : int
        Number of values used to scan fit stat profile
    scan_values : `~numpy.ndarray`
        Values to use for the scan.
    reoptimize : bool
        Re-optimize other free model parameters. Default is True.
    selection_optional : list of str
        Which additional quantities to estimate. Available options are:
            * "errn-errp": estimate asymmetric errors on parameter best fit value.
            * "ul": estimate upper limits.
            * "scan": estimate fit statistic profiles.

        By default all steps are executed.

    """

    tag = "ParameterEstimator"
    _available_selection_optional = ["errn-errp", "ul", "scan"]

    def __init__(
        self,
        n_sigma=1,
        n_sigma_ul=2,
        null_value=1e-150,
        scan_n_sigma=3,
        scan_min=None,
        scan_max=None,
        scan_n_values=30,
        scan_values=None,
        reoptimize=True,
        selection_optional="all",
    ):
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
        self.selection_optional = selection_optional
        self._fit = None

    def _setup_fit(self, datasets):
        # TODO: make fit stateless and configurable
        if self._fit is None or datasets is not self._fit.datasets:
            self._fit = Fit(datasets)

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
        self._setup_fit(datasets)
        result_fit = self._fit.run()

        return {
            f"{parameter.name}": parameter.value,
            "stat": result_fit.total_stat,
            "success": result_fit.success,
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

        with datasets.parameters.restore_values:

            # compute ts value
            parameter.value = self.null_value

            if self.reoptimize:
                parameter.frozen = True
                _ = self._fit.optimize()

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
        # TODO: make Fit stateless and configurable
        self._setup_fit(datasets)
        self._fit.optimize()

        res = self._fit.confidence(
            parameter=parameter, sigma=self.n_sigma, reoptimize=self.reoptimize
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
        self._setup_fit(datasets)
        self._fit.optimize()

        if self.scan_min and self.scan_max:
            bounds = (self.scan_min, self.scan_max)
        else:
            bounds = self.scan_n_sigma

        profile = self._fit.stat_profile(
            parameter=parameter,
            values=self.scan_values,
            bounds=bounds,
            nvalues=self.scan_n_values,
            reoptimize=self.reoptimize,
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
        self._setup_fit(datasets)
        self._fit.optimize()
        res = self._fit.confidence(
            parameter=parameter, sigma=self.n_sigma_ul, backend="scipy"
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

        with datasets.parameters.restore_values:

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
