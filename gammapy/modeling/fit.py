# Licensed under a 3-clause BSD style license - see LICENSE.rst
import itertools
import logging
import numpy as np
from gammapy.utils.pbar import progress_bar
from gammapy.utils.table import table_from_row_data
from .covariance import Covariance
from .iminuit import (
    confidence_iminuit,
    contour_iminuit,
    covariance_iminuit,
    optimize_iminuit,
)
from .scipy import confidence_scipy, optimize_scipy
from .sherpa import optimize_sherpa

__all__ = ["Fit"]

log = logging.getLogger(__name__)


class Registry:
    """Registry of available backends for given tasks.

    Gives users the power to extend from their scripts.
    Used by `Fit` below.

    Not sure if we should call it "backend" or "method" or something else.
    Probably we will code up some methods, e.g. for profile analysis ourselves,
    using scipy or even just Python / Numpy?
    """

    register = {
        "optimize": {
            "minuit": optimize_iminuit,
            "sherpa": optimize_sherpa,
            "scipy": optimize_scipy,
        },
        "covariance": {
            "minuit": covariance_iminuit,
            # "sherpa": covariance_sherpa,
            # "scipy": covariance_scipy,
        },
        "confidence": {
            "minuit": confidence_iminuit,
            # "sherpa": confidence_sherpa,
            "scipy": confidence_scipy,
        },
    }

    @classmethod
    def get(cls, task, backend):
        if task not in cls.register:
            raise ValueError(f"Unknown task {task!r}")

        backend_options = cls.register[task]

        if backend not in backend_options:
            raise ValueError(f"Unknown backend {backend!r} for task {task!r}")

        return backend_options[backend]


registry = Registry()


class Fit:
    """Fit class.

    The fit class provides a uniform interface to multiple fitting backends.
    Currently available: "minuit", "sherpa" and "scipy"

    Parameters
    ----------
    backend : {"minuit", "scipy" "sherpa"}
        Global backend used for fitting, default : minuit
    optimize_opts : dict
        Keyword arguments passed to the optimizer. For the `"minuit"` backend
        see https://iminuit.readthedocs.io/en/stable/reference.html#iminuit.Minuit
        for a detailed description of the available options. If there is an entry
        'migrad_opts', those options will be passed to `iminuit.Minuit.migrad()`.

        For the `"sherpa"` backend you can from the options:

            * `"simplex"`
            * `"levmar"`
            * `"moncar"`
            * `"gridsearch"`

        Those methods are described and compared in detail on
        http://cxc.cfa.harvard.edu/sherpa/methods/index.html. The available
        options of the optimization methods are described on the following
        pages in detail:

            * http://cxc.cfa.harvard.edu/sherpa/ahelp/neldermead.html
            * http://cxc.cfa.harvard.edu/sherpa/ahelp/montecarlo.html
            * http://cxc.cfa.harvard.edu/sherpa/ahelp/gridsearch.html
            * http://cxc.cfa.harvard.edu/sherpa/ahelp/levmar.html

        For the `"scipy"` backend the available options are described in detail here:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

    covariance_opts : dict
        Covariance options passed to the given backend.
    confidence_opts : dict
        Extra arguments passed to the backend. E.g. `iminuit.Minuit.minos` supports
        a ``maxcall`` option. For the scipy backend ``confidence_opts`` are forwarded
        to `~scipy.optimize.brentq`. If the confidence estimation fails, the bracketing
        interval can be adapted by modifying the the upper bound of the interval (``b``) value.
    store_trace : bool
        Whether to store the trace of the fit
    """

    def __init__(
        self,
        backend="minuit",
        optimize_opts=None,
        covariance_opts=None,
        confidence_opts=None,
        store_trace=False,
    ):
        self.store_trace = store_trace
        self.backend = backend

        if optimize_opts is None:
            optimize_opts = {"backend": backend}

        if covariance_opts is None:
            covariance_opts = {"backend": backend}

        if confidence_opts is None:
            confidence_opts = {"backend": backend}

        self.optimize_opts = optimize_opts
        self.covariance_opts = covariance_opts
        self.confidence_opts = confidence_opts
        self._minuit = None

    @property
    def minuit(self):
        """Iminuit object"""
        return self._minuit

    @staticmethod
    def _parse_datasets(datasets):
        from gammapy.datasets import Datasets

        datasets = Datasets(datasets)
        return datasets, datasets.parameters

    def run(self, datasets):
        """Run all fitting steps.

        Parameters
        ----------
        datasets : `Datasets` or list of `Dataset`
            Datasets to optimize.

        Returns
        -------
        fit_result : `FitResult`
            Fit result
        """
        optimize_result = self.optimize(datasets=datasets)

        if self.backend not in registry.register["covariance"]:
            log.warning("No covariance estimate - not supported by this backend.")
            return FitResult(optimize_result=optimize_result)

        covariance_result = self.covariance(datasets=datasets)

        return FitResult(
            optimize_result=optimize_result,
            covariance_result=covariance_result,
        )

    def optimize(self, datasets):
        """Run the optimization.

        Parameters
        ----------
        datasets : `Datasets` or list of `Dataset`
            Datasets to optimize.

        Returns
        -------
        optimize_result : `OptimizeResult`
            Optimization result
        """
        datasets, parameters = self._parse_datasets(datasets=datasets)
        datasets.parameters.check_limits()

        if len(parameters.free_parameters.names) == 0:
            raise ValueError("No free parameters for fitting")

        parameters.autoscale()

        kwargs = self.optimize_opts.copy()
        backend = kwargs.pop("backend", self.backend)

        compute = registry.get("optimize", backend)
        # TODO: change this calling interface!
        # probably should pass a fit statistic, which has a model, which has parameters
        # and return something simpler, not a tuple of three things
        factors, info, optimizer = compute(
            parameters=parameters,
            function=datasets.stat_sum,
            store_trace=self.store_trace,
            **kwargs,
        )

        if backend == "minuit":
            self._minuit = optimizer
            kwargs["method"] = "migrad"

        trace = table_from_row_data(info.pop("trace"))

        if self.store_trace:
            idx = [
                parameters.index(par)
                for par in parameters.unique_parameters.free_parameters
            ]
            unique_names = np.array(datasets.models.parameters_unique_names)[idx]
            trace.rename_columns(trace.colnames[1:], list(unique_names))

        # Copy final results into the parameters object
        parameters.set_parameter_factors(factors)
        parameters.check_limits()

        return OptimizeResult(
            models=datasets.models.copy(),
            total_stat=datasets.stat_sum(),
            backend=backend,
            method=kwargs.get("method", backend),
            trace=trace,
            **info,
        )

    def covariance(self, datasets):
        """Estimate the covariance matrix.

        Assumes that the model parameters are already optimised.

        Parameters
        ----------
        datasets : `Datasets` or list of `Dataset`
            Datasets to optimize.

        Returns
        -------
        result : `CovarianceResult`
            Results
        """
        datasets, unique_pars = self._parse_datasets(datasets=datasets)
        parameters = datasets.models.parameters

        kwargs = self.covariance_opts.copy()
        kwargs["minuit"] = self.minuit
        backend = kwargs.pop("backend", self.backend)
        compute = registry.get("covariance", backend)

        with unique_pars.restore_status():
            if self.backend == "minuit":
                method = "hesse"
            else:
                method = ""

            factor_matrix, info = compute(
                parameters=unique_pars, function=datasets.stat_sum, **kwargs
            )

            datasets.models.covariance = Covariance.from_factor_matrix(
                parameters=parameters, matrix=factor_matrix
            )

        # TODO: decide what to return, and fill the info correctly!
        return CovarianceResult(
            backend=backend,
            method=method,
            success=info["success"],
            message=info["message"],
            matrix=datasets.models.covariance.data.copy(),
        )

    def confidence(self, datasets, parameter, sigma=1, reoptimize=True):
        """Estimate confidence interval.

        Extra ``kwargs`` are passed to the backend.
        E.g. `iminuit.Minuit.minos` supports a ``maxcall`` option.

        For the scipy backend ``kwargs`` are forwarded to `~scipy.optimize.brentq`. If the
        confidence estimation fails, the bracketing interval can be adapted by modifying the
        the upper bound of the interval (``b``) value.

        Parameters
        ----------
        datasets : `Datasets` or list of `Dataset`
            Datasets to optimize.
        parameter : `~gammapy.modeling.Parameter`
            Parameter of interest
        sigma : float
            Number of standard deviations for the confidence level
        reoptimize : bool
            Re-optimize other parameters, when computing the confidence region.

        Returns
        -------
        result : dict
            Dictionary with keys "errp", 'errn", "success" and "nfev".
        """
        datasets, parameters = self._parse_datasets(datasets=datasets)

        kwargs = self.confidence_opts.copy()
        backend = kwargs.pop("backend", self.backend)

        compute = registry.get("confidence", backend)
        parameter = parameters[parameter]

        with parameters.restore_status():
            result = compute(
                parameters=parameters,
                parameter=parameter,
                function=datasets.stat_sum,
                sigma=sigma,
                reoptimize=reoptimize,
                **kwargs,
            )

        result["errp"] *= parameter.scale
        result["errn"] *= parameter.scale
        return result

    def stat_profile(self, datasets, parameter, reoptimize=False):
        """Compute fit statistic profile.

        The method used is to vary one parameter, keeping all others fixed.
        So this is taking a "slice" or "scan" of the fit statistic.

        Parameters
        ----------
        datasets : `Datasets` or list of `Dataset`
            Datasets to optimize.
        parameter : `~gammapy.modeling.Parameter`
            Parameter of interest. The specification for the scan, such as bounds
            and number of values is taken from the parameter object.
        reoptimize : bool
            Re-optimize other parameters, when computing the confidence region.

        Returns
        -------
        results : dict
            Dictionary with keys "values", "stat" and "fit_results". The latter contains an
            empty list, if `reoptimize` is set to False
        """
        datasets, parameters = self._parse_datasets(datasets=datasets)
        parameter = parameters[parameter]
        values = parameter.scan_values

        stats = []
        fit_results = []
        with parameters.restore_status():
            for value in progress_bar(values, desc="Scan values"):
                parameter.value = value
                if reoptimize:
                    parameter.frozen = True
                    result = self.optimize(datasets=datasets)
                    stat = result.total_stat
                    fit_results.append(result)
                else:
                    stat = datasets.stat_sum()
                stats.append(stat)

        return {
            f"{parameter.name}_scan": values,
            "stat_scan": np.array(stats),
            "fit_results": fit_results,
        }

    def stat_surface(self, datasets, x, y, reoptimize=False):
        """Compute fit statistic surface.

        The method used is to vary two parameters, keeping all others fixed.
        So this is taking a "slice" or "scan" of the fit statistic.

        Caveat: This method can be very computationally intensive and slow

        See also: `Fit.stat_contour`

        Parameters
        ----------
        datasets : `Datasets` or list of `Dataset`
            Datasets to optimize.
        x, y : `~gammapy.modeling.Parameter`
            Parameters of interest
        reoptimize : bool
            Re-optimize other parameters, when computing the confidence region.

        Returns
        -------
        results : dict
            Dictionary with keys "x_values", "y_values", "stat" and "fit_results".
            The latter contains an empty list, if `reoptimize` is set to False
        """
        datasets, parameters = self._parse_datasets(datasets=datasets)

        x, y = parameters[x], parameters[y]

        stats = []
        fit_results = []

        with parameters.restore_status():
            for x_value, y_value in progress_bar(
                itertools.product(x.scan_values, y.scan_values), desc="Trial values"
            ):
                x.value, y.value = x_value, y_value

                if reoptimize:
                    x.frozen, y.frozen = True, True
                    result = self.optimize(datasets=datasets)
                    stat = result.total_stat
                    fit_results.append(result)
                else:
                    stat = datasets.stat_sum()

                stats.append(stat)

        shape = (len(x.scan_values), len(y.scan_values))
        stats = np.array(stats).reshape(shape)

        if reoptimize:
            fit_results = np.array(fit_results).reshape(shape)

        return {
            f"{x.name}_scan": x.scan_values,
            f"{y.name}_scan": y.scan_values,
            "stat_scan": stats,
            "fit_results": fit_results,
        }

    def stat_contour(self, datasets, x, y, numpoints=10, sigma=1):
        """Compute stat contour.

        Calls ``iminuit.Minuit.mncontour``.

        This is a contouring algorithm for a 2D function
        which is not simply the fit statistic function.
        That 2D function is given at each point ``(par_1, par_2)``
        by re-optimising all other free parameters,
        and taking the fit statistic at that point.

        Very compute-intensive and slow.

        Parameters
        ----------
        datasets : `Datasets` or list of `Dataset`
            Datasets to optimize.
        x, y : `~gammapy.modeling.Parameter`
            Parameters of interest
        numpoints : int
            Number of contour points
        sigma : float
            Number of standard deviations for the confidence level

        Returns
        -------
        result : dict
            Dictionary containing the parameter values defining the contour, with the
            boolean flag "success" and the info objects from ``mncontour``.
        """
        datasets, parameters = self._parse_datasets(datasets=datasets)

        x = parameters[x]
        y = parameters[y]

        with parameters.restore_status():
            result = contour_iminuit(
                parameters=parameters,
                function=datasets.stat_sum,
                x=x,
                y=y,
                numpoints=numpoints,
                sigma=sigma,
            )

        x_name = x.name
        y_name = y.name
        x = result["x"] * x.scale
        y = result["y"] * y.scale

        return {
            x_name: x,
            y_name: y,
            "success": result["success"],
        }


class FitStepResult:
    """Fit result base class"""

    def __init__(self, backend, method, success, message):
        self._success = success
        self._message = message
        self._backend = backend
        self._method = method

    @property
    def backend(self):
        """Optimizer backend used for the fit."""
        return self._backend

    @property
    def method(self):
        """Optimizer method used for the fit."""
        return self._method

    @property
    def success(self):
        """Fit success status flag."""
        return self._success

    @property
    def message(self):
        """Optimizer status message."""
        return self._message

    def __repr__(self):
        return (
            f"{self.__class__.__name__}\n\n"
            f"\tbackend    : {self.backend}\n"
            f"\tmethod     : {self.method}\n"
            f"\tsuccess    : {self.success}\n"
            f"\tmessage    : {self.message}\n"
        )


class CovarianceResult(FitStepResult):
    """Covariance result object."""

    def __init__(self, matrix=None, **kwargs):
        self._matrix = matrix
        super().__init__(**kwargs)

    @property
    def matrix(self):
        """Covariance matrix (`~numpy.ndarray`)"""
        return self._matrix


class OptimizeResult(FitStepResult):
    """Optimize result object."""

    def __init__(self, models, nfev, total_stat, trace, **kwargs):
        self._models = models
        self._nfev = nfev
        self._total_stat = total_stat
        self._trace = trace
        super().__init__(**kwargs)

    @property
    def parameters(self):
        """Best fit parameters"""
        return self.models.parameters

    @property
    def models(self):
        """Best fit models"""
        return self._models

    @property
    def trace(self):
        """Parameter trace from the optimisation"""
        return self._trace

    @property
    def nfev(self):
        """Number of function evaluations."""
        return self._nfev

    @property
    def total_stat(self):
        """Value of the fit statistic at minimum."""
        return self._total_stat

    def __repr__(self):
        str_ = super().__repr__()
        str_ += f"\tnfev       : {self.nfev}\n"
        str_ += f"\ttotal stat : {self.total_stat:.2f}\n\n"
        return str_


class FitResult:
    """Fit result class

    Parameters
    ----------
    optimize_result : `OptimizeResult`
        Result of the optimization step.
    covariance_result : `CovarianceResult`
        Result of the covariance step.
    """

    def __init__(self, optimize_result=None, covariance_result=None):
        self._optimize_result = optimize_result

        if covariance_result:
            self.optimize_result.models.covariance = covariance_result.matrix

        self._covariance_result = covariance_result

    # TODO: is the convenience access needed?
    @property
    def parameters(self):
        """Best fit parameters of the optimization step"""
        return self.optimize_result.parameters

    # TODO: is the convenience access needed?
    @property
    def models(self):
        """Best fit parameters of the optimization step"""
        return self.optimize_result.models

    # TODO: is the convenience access needed?
    @property
    def total_stat(self):
        """Total stat of the optimization step"""
        return self.optimize_result.total_stat

    # TODO: is the convenience access needed?
    @property
    def trace(self):
        """Parameter trace of the optimisation step"""
        return self.optimize_result.trace

    # TODO: is the convenience access needed?
    @property
    def nfev(self):
        """Number of function evaluations of the optimisation step"""
        return self.optimize_result.nfev

    # TODO: is the convenience access needed?
    @property
    def backend(self):
        """Optimizer backend used for the fit."""
        return self.optimize_result.backend

    # TODO: is the convenience access needed?
    @property
    def method(self):
        """Optimizer method used for the fit."""
        return self.optimize_result.method

    # TODO: is the convenience access needed?
    @property
    def message(self):
        """Optimizer status message."""
        return self.optimize_result.message

    @property
    def success(self):
        """Total success flag"""
        success = self.optimize_result.success

        if self.covariance_result:
            success &= self.covariance_result.success

        return success

    @property
    def optimize_result(self):
        """Optimize result"""
        return self._optimize_result

    @property
    def covariance_result(self):
        """Optimize result"""
        return self._covariance_result

    def __repr__(self):
        str_ = ""
        if self.optimize_result:
            str_ += str(self.optimize_result)

        if self.covariance_result:
            str_ += str(self.covariance_result)

        return str_
