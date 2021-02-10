# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from gammapy.datasets import Datasets
from gammapy.modeling import Fit, stat_profile_ul_scipy
from .core import Estimator
from gammapy.utils.interpolation import interpolation_scale

log = logging.getLogger(__name__)

def make_scan_values(
    parameter,
    bounds=3,
    nvalues=30,
    err_rel_min=0.05,
    scaling='lin'
):
    """Prepare values scan

    Parameters
    ----------
    parameter : `Parameter`
        For which parameter to get the values
    bounds : int or tuple of float
        When an `int` is passed the bounds are computed from `bounds * sigma`
        from the best fit value of the parameter, where `sigma` corresponds to
        the one sigma error on the parameter. If a tuple of floats is given
        those are taken as the min and max values and ``nvalues`` are generated
        between those.
    nvalues : int
        Number of parameter grid points to use.
    err_rel_min : float
       Minimun relative error allowed (default is 5%).
       If the relative error if lower than `err_rel_min` or 
       if the parameter error is not defined, then the parameter
       error is condider to be the parameter value.
       Used only when an `int` is passed as `bounds`.
    scaling: {'lin', 'log', 'sqrt'}
        Choose values scaling. Defauld is linear ('lin')
    
    Returns
    -------
    results : np.array
        values
    """

    if isinstance(bounds, tuple):
        parmin, parmax = bounds
    else:
        parmin, parmax = make_scan_bounds(parameter, bounds, err_rel_min)
    scaler = interpolation_scale(scaling)
    parmin, parmax = scaler(parmin, parmax )
    values = np.linspace(parmin, parmax, nvalues)
    return scaler.inverse(values)

def make_scan_bounds(
    parameter,
    scan_n_sigma=3,
    err_rel_min=0.05,
):
    """Prepare values scan bounds

    Parameters
    ----------
    parameter : `Parameter`
        For which parameter to get the values
    scan_n_sigma : int
        The bounds are computed from `scan_n_sigma * sigma`
        from the best fit value of the parameter, where `sigma` corresponds to
        the one sigma error on the parameter.
    err_rel_min : float
       Minimun relative error allowed (default is 5%).
       If the relative error if lower than `err_rel_min` or 
       if the parameter error is not defined, then the parameter
       error is condider to be the parameter value.
       Used only when an `int` is passed as `bounds`.
    
    Returns
    -------
    results : np.array
        values
    """
    parval = parameter.value
    parerr = parameter.error
    err_rel = np.abs(parameter.error/parameter.value)
    if np.isnan(parerr) or (err_rel < err_rel_min):
        parerr = np.abs(parval)
    return [parval - scan_n_sigma * parerr, np.abs(parval) + scan_n_sigma * parerr]

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
    scan_values : `~numpy.ndarray`
        Values to use for the scan.
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
        Re-optimize other free model parameters. Default is True.
    selection_optional : list of str
        Which additional quantities to estimate. Available options are:

            * "all": all the optional steps are executed
            * "errn-errp": estimate asymmetric errors on parameter best fit value.
            * "ul": estimate upper limits.
            * "scan": estimate fit statistic profiles.

        Default is None so the optionnal steps are not executed.


    """

    tag = "ParameterEstimator"
    _available_selection_optional = ["errn-errp", "ul", "scan"]

    def __init__(
        self,
        n_sigma=1,
        n_sigma_ul=2,
        null_value=1e-150,
        scan_values=None,
        backend="minuit",
        optimize_opts=None,
        covariance_opts=None,
        reoptimize=True,
        selection_optional=None,
    ):
        
        self.n_sigma = n_sigma
        self.n_sigma_ul = n_sigma_ul
        self.null_value = null_value

        # scan parameters
        self.scan_values = scan_values

        self.backend = backend
        if optimize_opts is None:
            optimize_opts = {}
        if covariance_opts is None:
            covariance_opts = {}
        self.optimize_opts = optimize_opts
        self.covariance_opts = covariance_opts

        self.reoptimize = reoptimize
        self.selection_optional = selection_optional
        self._fit = None
        self._profile = None

    def fit(self, datasets):
        if self._fit is None or datasets is not self._fit.datasets:
            self._fit = Fit(
                datasets,
                backend=self.backend,
                optimize_opts=self.optimize_opts,
                covariance_opts=self.covariance_opts,
            )
        return self._fit

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
        self.fit(datasets)
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

        with datasets.parameters.restore_status():

            # compute ts value
            parameter.value = self.null_value

            if self.reoptimize:
                parameter.frozen = True
                _ = self._fit.optimize(**self.optimize_opts)

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
        self.fit(datasets)
        self._fit.optimize(**self.optimize_opts)

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
        self.fit(datasets)
        self._fit.optimize(**self.optimize_opts)

        if self.scan_values is None:
            self.scan_values = make_scan_values(parameter)
        profile = self._fit.stat_profile(
            parameter=parameter,
            values=self.scan_values,
            reoptimize=self.reoptimize,
        )
        self._profile = {
            f"{parameter.name}_scan": profile[f"{parameter.name}_scan"],
            "stat_scan": profile["stat_scan"],
        }
        return self._profile

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
            
        if self.ul_method == "confidence":
            self.fit(datasets)
            self._fit.optimize(**self.optimize_opts)
            res = self._fit.confidence(parameter=parameter, sigma=self.n_sigma_ul)
            ul = {f"{parameter.name}_ul": res["errp"] + parameter.value}
        elif self.ul_method == "profile":
            if self.scan_values is None:
                self.scan_values = make_scan_values(parameter)
            if self._profile is None:
                profile = self.estimate_scan(self, datasets, parameter)
            ul = stat_profile_ul_scipy(self.scan_values, profile["stat_scan"], delta_ts=4, interp_scale="sqrt")
        return ul

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
