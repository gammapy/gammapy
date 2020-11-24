# Licensed under a 3-clause BSD style license - see LICENSE.rst
import itertools
import logging
import numpy as np
from astropy.utils import lazyproperty
from gammapy.utils.table import table_from_row_data
from .covariance import Covariance
from .iminuit import confidence_iminuit, covariance_iminuit, mncontour, optimize_iminuit
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
    datasets : `Datasets`
        Datasets
    """

    def __init__(self, datasets, store_trace=False):
        from gammapy.datasets import Datasets

        self.store_trace = store_trace
        self.datasets = Datasets(datasets)

    @lazyproperty
    def _parameters(self):
        return self.datasets.parameters

    @lazyproperty
    def _models(self):
        return self.datasets.models

    def run(self, backend="minuit", optimize_opts=None, covariance_opts=None):
        """
        Run all fitting steps.

        Parameters
        ----------
        backend : str
            Backend used for fitting, default : minuit
        optimize_opts : dict
            Options passed to `Fit.optimize`.
        covariance_opts : dict
            Options passed to `Fit.covariance`.

        Returns
        -------
        fit_result : `FitResult`
            Results
        """

        if optimize_opts is None:
            optimize_opts = {}
        optimize_result = self.optimize(backend, **optimize_opts)

        if covariance_opts is None:
            covariance_opts = {}

        if backend not in registry.register["covariance"]:
            log.warning("No covariance estimate - not supported by this backend.")
            return optimize_result

        covariance_result = self.covariance(backend, **covariance_opts)
        # TODO: not sure how best to report the results
        # back or how to form the FitResult object.
        optimize_result._success = optimize_result.success and covariance_result.success

        return optimize_result

    def optimize(self, backend="minuit", **kwargs):
        """Run the optimization.

        Parameters
        ----------
        backend : str
            Which backend to use (see ``gammapy.modeling.registry``)
        **kwargs : dict
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

            For the `"scipy"` backend the available options are desribed in detail here:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

        Returns
        -------
        fit_result : `FitResult`
            Results
        """
        parameters = self._parameters
        parameters.check_limits()

        # TODO: expose options if / when to scale? On the Fit class?
        if np.all(self._models.covariance.data == 0):
            parameters.autoscale()

        compute = registry.get("optimize", backend)
        # TODO: change this calling interface!
        # probably should pass a fit statistic, which has a model, which has parameters
        # and return something simpler, not a tuple of three things
        factors, info, optimizer = compute(
            parameters=parameters,
            function=self.datasets.stat_sum,
            store_trace=self.store_trace,
            **kwargs,
        )

        # TODO: Change to a stateless interface for minuit also, or if we must support
        # stateful backends, put a proper, backend-agnostic solution for this.
        # As preliminary solution would like to provide a possibility that the user
        # can access the Minuit object, because it features a lot useful functionality
        if backend == "minuit":
            self.minuit = optimizer

        trace = table_from_row_data(info.pop("trace"))

        if self.store_trace:
            pars = self._models.parameters
            idx = [pars.index(par) for par in pars.unique_parameters.free_parameters]
            unique_names = np.array(self._models.parameters_unique_names)[idx]
            trace.rename_columns(trace.colnames[1:], list(unique_names))

        # Copy final results into the parameters object
        parameters.set_parameter_factors(factors)
        parameters.check_limits()
        return OptimizeResult(
            parameters=parameters,
            total_stat=self.datasets.stat_sum(),
            backend=backend,
            method=kwargs.get("method", backend),
            trace=trace,
            **info,
        )

    def covariance(self, backend="minuit", **kwargs):
        """Estimate the covariance matrix.

        Assumes that the model parameters are already optimised.

        Parameters
        ----------
        backend : str
            Which backend to use (see ``gammapy.modeling.registry``)

        Returns
        -------
        result : `CovarianceResult`
            Results
        """
        compute = registry.get("covariance", backend)
        parameters = self._parameters

        # TODO: wrap MINUIT in a stateless backend
        with parameters.restore_values:
            if backend == "minuit":
                method = "hesse"
                if hasattr(self, "minuit"):
                    factor_matrix, info = compute(self.minuit)
                else:
                    raise RuntimeError("To use minuit, you must first optimize.")
            else:
                method = ""
                factor_matrix, info = compute(
                    parameters, self.datasets.stat_sum, **kwargs
                )

            covariance = Covariance.from_factor_matrix(
                parameters=self._models.parameters, matrix=factor_matrix
            )
            self._models.covariance = covariance

        # TODO: decide what to return, and fill the info correctly!
        return CovarianceResult(
            backend=backend,
            method=method,
            parameters=parameters,
            success=info["success"],
            message=info["message"],
        )

    def confidence(
        self, parameter, backend="minuit", sigma=1, reoptimize=True, **kwargs
    ):
        """Estimate confidence interval.

        Extra ``kwargs`` are passed to the backend.
        E.g. `iminuit.Minuit.minos` supports a ``maxcall`` option.

        For the scipy backend ``kwargs`` are forwarded to `~scipy.optimize.brentq`. If the
        confidence estimation fails, the bracketing interval can be adapted by modifying the
        the upper bound of the interval (``b``) value.

        Parameters
        ----------
        backend : str
            Which backend to use (see ``gammapy.modeling.registry``)
        parameter : `~gammapy.modeling.Parameter`
            Parameter of interest
        sigma : float
            Number of standard deviations for the confidence level
        reoptimize : bool
            Re-optimize other parameters, when computing the confidence region.
        **kwargs : dict
            Keyword argument passed ot the confidence estimation method.

        Returns
        -------
        result : dict
            Dictionary with keys "errp", 'errn", "success" and "nfev".
        """
        compute = registry.get("confidence", backend)
        parameters = self._parameters
        parameter = parameters[parameter]

        # TODO: wrap MINUIT in a stateless backend
        with parameters.restore_values:
            if backend == "minuit":
                if hasattr(self, "minuit"):
                    # This is ugly. We will access parameters and make a copy
                    # from the backend, to avoid modifying the state
                    result = compute(
                        self.minuit, parameters, parameter, sigma, **kwargs
                    )
                else:
                    raise RuntimeError("To use minuit, you must first optimize.")
            else:
                result = compute(
                    parameters,
                    parameter,
                    self.datasets.stat_sum,
                    sigma,
                    reoptimize,
                    **kwargs,
                )

        result["errp"] *= parameter.scale
        result["errn"] *= parameter.scale
        return result

    def stat_profile(
        self,
        parameter,
        values=None,
        bounds=2,
        nvalues=11,
        reoptimize=False,
        optimize_opts=None,
    ):
        """Compute fit statistic profile.

        The method used is to vary one parameter, keeping all others fixed.
        So this is taking a "slice" or "scan" of the fit statistic.

        See also: `Fit.minos_profile`.

        Parameters
        ----------
        parameter : `~gammapy.modeling.Parameter`
            Parameter of interest
        values : `~astropy.units.Quantity` (optional)
            Parameter values to evaluate the fit statistic for.
        bounds : int or tuple of float
            When an `int` is passed the bounds are computed from `bounds * sigma`
            from the best fit value of the parameter, where `sigma` corresponds to
            the one sigma error on the parameter. If a tuple of floats is given
            those are taken as the min and max values and ``nvalues`` are linearly
            spaced between those.
        nvalues : int
            Number of parameter grid points to use.
        reoptimize : bool
            Re-optimize other parameters, when computing the fit statistic profile.

        Returns
        -------
        results : dict
            Dictionary with keys "values", "stat" and "fit_results". The latter contains an
            empty list, if `reoptimize` is set to False
        """
        parameters = self._parameters
        parameter = parameters[parameter]

        optimize_opts = optimize_opts or {}

        if values is None:
            if isinstance(bounds, tuple):
                parmin, parmax = bounds
            else:
                if np.isnan(parameter.error):
                    raise ValueError("Parameter error is not properly set.")
                parerr = parameter.error
                parval = parameter.value
                parmin, parmax = parval - bounds * parerr, parval + bounds * parerr

            values = np.linspace(parmin, parmax, nvalues)

        stats = []
        fit_results = []
        with parameters.restore_values:
            for value in values:
                parameter.value = value
                if reoptimize:
                    parameter.frozen = True
                    result = self.optimize(**optimize_opts)
                    stat = result.total_stat
                    fit_results.append(result)
                else:
                    stat = self.datasets.stat_sum()
                stats.append(stat)

        return {
            f"{parameter.name}_scan": values,
            "stat_scan": np.array(stats),
            "fit_results": fit_results,
        }

    def stat_surface(self, x, y, x_values, y_values, reoptimize=False, **optimize_opts):
        """Compute fit statistic surface.

        The method used is to vary two parameters, keeping all others fixed.
        So this is taking a "slice" or "scan" of the fit statistic.

        Caveat: This method can be very computationally intensive and slow

        See also: `Fit.minos_contour`

        Parameters
        ----------
        x, y : `~gammapy.modeling.Parameter`
            Parameters of interest
        x_values, y_values : list or `numpy.ndarray`
            Parameter values to evaluate the fit statistic for.
        reoptimize : bool
            Re-optimize other parameters, when computing the fit statistic profile.
        **optimize_opts : dict
            Keyword arguments passed to the optimizer. See `Fit.optimize` for further details.

        Returns
        -------
        results : dict
            Dictionary with keys "x_values", "y_values", "stat" and "fit_results". The latter contains an
            empty list, if `reoptimize` is set to False
        """
        parameters = self._parameters
        x = parameters[x]
        y = parameters[y]

        stats = []
        fit_results = []
        with parameters.restore_values:
            for x_value, y_value in itertools.product(x_values, y_values):
                # TODO: Remove log.info() and provide a nice progress bar
                log.info(f"Processing: x={x_value}, y={y_value}")
                x.value = x_value
                y.value = y_value
                if reoptimize:
                    x.frozen = True
                    y.frozen = True
                    result = self.optimize(**optimize_opts)
                    stat = result.total_stat
                    fit_results.append(result)
                else:
                    stat = self.datasets.stat_sum()

                stats.append(stat)

        shape = (np.asarray(x_values).shape[0], np.asarray(y_values).shape[0])
        stats = np.array(stats)
        stats = stats.reshape(shape)
        if reoptimize:
            fit_results = np.array(fit_results)
            fit_results = fit_results.reshape(shape)

        return {
            f"{x.name}_scan": x_values,
            f"{y.name}_scan": y_values,
            "stat_scan": stats,
            "fit_results": fit_results,
        }

    def minos_contour(self, x, y, numpoints=10, sigma=1.0):
        """Compute MINOS contour.

        Calls ``iminuit.Minuit.mncontour``.

        This is a contouring algorithm for a 2D function
        which is not simply the fit statistic function.
        That 2D function is given at each point ``(par_1, par_2)``
        by re-optimising all other free parameters,
        and taking the fit statistic at that point.

        Very compute-intensive and slow.

        Parameters
        ----------
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
        parameters = self._parameters
        x = parameters[x]
        y = parameters[y]

        with parameters.restore_values:
            result = mncontour(self.minuit, parameters, x, y, numpoints, sigma)

        x_name = x.name
        y_name = y.name
        x = result["x"] * x.scale
        y = result["y"] * y.scale

        return {
            x_name: x,
            y_name: y,
            "success": result["success"],
            f"{x_name}_info": result["x_info"],
            f"{y_name}_info": result["y_info"],
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

    def __init__(self, nfev, total_stat, trace, **kwargs):
        self._nfev = nfev
        self._total_stat = total_stat
        self._trace = trace
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

    def __repr__(self):
        str_ = super().__repr__()
        str_ += f"\tnfev       : {self.nfev}\n"
        str_ += f"\ttotal stat : {self.total_stat:.2f}\n"
        return str_
