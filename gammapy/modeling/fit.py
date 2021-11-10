# Licensed under a 3-clause BSD style license - see LICENSE.rst
import itertools
import logging
import numpy as np
from gammapy.utils.table import table_from_row_data
from gammapy.utils.pbar import progress_bar
from .covariance import Covariance
from .iminuit import confidence_iminuit, covariance_iminuit, contour_iminuit, optimize_iminuit
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
        see https://iminuit.readthedocs.io/en/latest/api.html#iminuit.Minuit
        for a detailed description of the available options. If there is an entry
        'migrad_opts', those options will be passed to `iminuit.Minuit.migrad()`.

        For the `"sherpa"` backend you can from the options `method = {"simplex",  "levmar", "moncar", "gridsearch"}`
        Those methods are described and compared in detail on
        http://cxc.cfa.harvard.edu/sherpa/methods/index.html. The available
        options of the optimization methods are described on the following
        pages in detail:
            * http://cxc.cfa.harvard.edu/sherpa/ahelp/neldermead.html
            * http://cxc.cfa.harvard.edu/sherpa/ahelp/montecarlo.html
            * http://cxc.cfa.harvard.edu/sherpa/ahelp/gridsearch.html
            * http://cxc.cfa.harvard.edu/sherpa/ahelp/levmar.html

        For the `"scipy"` backend the available options are decsribed in detail here:
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

        Returns
        -------
        fit_result : `FitResult`
            Results
        """
        optimize_result = self.optimize(datasets=datasets)

        if self.backend not in registry.register["covariance"]:
            log.warning("No covariance estimate - not supported by this backend.")
            return optimize_result

        covariance_result = self.covariance(datasets=datasets)

        optimize_result._covariance_result = covariance_result

        return optimize_result

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
            parameters=parameters,
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
        datasets, parameters = self._parse_datasets(datasets=datasets)

        kwargs = self.covariance_opts.copy()
        backend = kwargs.pop("backend", self.backend)
        compute = registry.get("covariance", backend)

        with parameters.restore_status():
            if self.backend == "minuit":
                method = "hesse"
            else:
                method = ""

            factor_matrix, info = compute(
                parameters=parameters, function=datasets.stat_sum, **kwargs
            )

            datasets.models.covariance = Covariance.from_factor_matrix(
                parameters=parameters, matrix=factor_matrix
            )

        # TODO: decide what to return, and fill the info correctly!
        return CovarianceResult(
            parameters=parameters,
            backend=backend,
            method=method,
            success=info["success"],
            message=info["message"],
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
            Dictionary with keys "x_values", "y_values", "stat" and "fit_results". The latter contains an
            empty list, if `reoptimize` is set to False
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
                sigma=sigma
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


class FitResult:
    """Fit result base class"""

    def __init__(self, parameters, backend, method, success, message):
        self._parameters = parameters
        self._success = success
        self._message = message
        self._backend = backend
        self._method = method

    @property
    def parameters(self):
        """Optimizer backend used for the fit."""
        return self._parameters

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


class CovarianceResult(FitResult):
    """Covariance result object."""

    pass


class OptimizeResult(FitResult):
    """Optimize result object."""

    def __init__(self, nfev, total_stat, trace, covariance_result=None, **kwargs):
        self._nfev = nfev
        self._total_stat = total_stat
        self._trace = trace
        self._covariance_result = covariance_result
        super().__init__(**kwargs)

    @property
    def trace(self):
        """Optimizer backend used for the fit."""
        return self._trace

    @property
    def nfev(self):
        """Number of function evaluations."""
        return self._nfev

    @property
    def total_stat(self):
        """Value of the fit statistic at minimum."""
        return self._total_stat

    @property
    def covariance_result(self):
        """Covariance results."""
        return self._covariance_result

    def __repr__(self):
        str_ = super().__repr__()
        str_ += f"\tnfev       : {self.nfev}\n"
        str_ += f"\ttotal stat : {self.total_stat:.2f}\n"
        if self.covariance_result is not None:
            str_ += self.covariance_result.__repr__()
        return str_
