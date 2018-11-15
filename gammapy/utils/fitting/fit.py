from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import abc
import numpy as np
from astropy.utils.misc import InheritDocstrings
from ...extern import six
from .iminuit import optimize_iminuit, covar_iminuit, confidence_iminuit
from .sherpa import optimize_sherpa
from .scipy import optimize_scipy, covar_scipy

__all__ = ["Fit"]

log = logging.getLogger(__name__)


class FitMeta(InheritDocstrings, abc.ABCMeta):
    pass


class Registry(object):
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
        "covar": {
            "minuit": covar_iminuit,
            # "scipy": covar_scipy,
        },
        "confidence": {"minuit": confidence_iminuit},
    }

    @classmethod
    def get(cls, task, backend):
        if task not in cls.register:
            raise ValueError("Unknown task {!r}".format(task))

        task = cls.register[task]

        if backend not in task:
            raise ValueError("Unknown method {!r} for task {!r}".format(task, backend))

        return task[backend]


registry = Registry()


@six.add_metaclass(FitMeta)
class Fit(object):
    """Abstract Fit base class.
    """

    @abc.abstractmethod
    def total_stat(self, parameters):
        """Total likelihood given the current model parameters"""
        pass

    def run(self, optimize_opts=None, covar_opts=None):
        """
        Run all fitting steps.

        Parameters
        ----------
        optimize_opts : dict
            Options passed to `Fit.optimize`.
        covar_opts : dict
            Options passed to `Fit.covar`.

        Returns
        -------
        fit_result : `FitResult`
            Results
        """
        if optimize_opts is None:
            optimize_opts = {}
        optimize_result = self.optimize(**optimize_opts)

        if covar_opts is None:
            covar_opts = {}

        covar_opts.setdefault("backend", "minuit")
        if covar_opts["backend"] in registry.register["covar"]:
            covar_result = self.covar(**covar_opts)
            # TODO: not sure how best to report the results
            # back or how to form the FitResult object.
            optimize_result._model = covar_result.model
            optimize_result._success = optimize_result.success and covar_result.success
            optimize_result._nfev += covar_result.nfev
        else:
            log.warning("No covar estimate - not supported by this backend.")

        return optimize_result

    def optimize(self, backend="minuit", **kwargs):
        """Run the optimization.

        Parameters
        ----------
        backend : {"minuit", "sherpa"}
            Which fitting backend to use.
        **kwargs : dict
            Keyword arguments passed to the optimizer. For the `"minuit"` backend
            see https://iminuit.readthedocs.io/en/latest/api.html#iminuit.Minuit
            for a detailed description of the available options. For the `"sherpa"`
            backend you can from the options `method = {"simplex",  "levmar", "moncar", "gridsearch"}`
            Those methods are described and compared in detail on
            http://cxc.cfa.harvard.edu/sherpa/methods/index.html. The available
            options of the optimization methods are described on the following
            pages in detail:

                * http://cxc.cfa.harvard.edu/sherpa/ahelp/neldermead.html
                * http://cxc.cfa.harvard.edu/sherpa/ahelp/montecarlo.html
                * http://cxc.cfa.harvard.edu/sherpa/ahelp/gridsearch.html
                * http://cxc.cfa.harvard.edu/sherpa/ahelp/levmar.html


        Returns
        -------
        fit_result : `FitResult`
            Results
        """
        parameters = self._model.parameters

        if parameters.apply_autoscale:
            parameters.autoscale()

        compute = registry.get("optimize", backend)
        # TODO: change this calling interface!
        # probably should pass a likelihood, which has a model, which has parameters
        # and return something simpler, not a tuple of three things
        factors, info, optimizer = compute(
            parameters=parameters, function=self.total_stat, **kwargs
        )

        # TODO: Change to a stateless interface for minuit also, or if we must support
        # stateful backends, put a proper, backend-agnostic solution for this.
        # As preliminary solution would like to provide a possibility that the user
        # can access the Minuit object, because it features a lot useful functionality
        if backend == "minuit":
            self.minuit = optimizer

        # Copy final results into the parameters object
        parameters.set_parameter_factors(factors)

        return FitResult(
            model=self._model.copy(),
            total_stat=self.total_stat(self._model.parameters),
            backend=backend,
            method=kwargs.get("method", backend),
            **info
        )

    def covar(self, backend="minuit"):
        """Estimate the covariance matrix.

        Assumes that the model parameters are already optimised.

        Returns
        -------
        result : `CovarResult`
            Results
        """
        compute = registry.get("covar", backend)
        parameters = self._model.parameters

        # TODO: wrap MINUIT in a stateless backend
        if backend == "minuit":
            if hasattr(self, "minuit"):
                covariance_factors = compute(self.minuit)
            else:
                raise RuntimeError("To use minuit, you must first optimize.")
        else:
            function = self.total_stat
            covariance_factors = compute(parameters, function)

        parameters.set_covariance_factors(covariance_factors)

        # TODO: decide what to return, and fill the info correctly!
        return CovarResult(model=self._model.copy(), success=True, nfev=0)

    def confidence(self, parameter, backend="minuit", sigma=1, maxcall=0):
        """Estimate confidence interval.

        Returns
        -------
        result : dict
            Results
        """
        compute = registry.get("confidence", backend)
        parameters = self._model.parameters

        # TODO: wrap MINUIT in a stateless backend
        if backend == "minuit":
            if hasattr(self, "minuit"):
                result = compute(
                    self.minuit, parameters, parameter, sigma, maxcall
                )
                # TODO: decide about result format
                return result
            else:
                raise RuntimeError("To use minuit, you must first optimize.")
        else:
            raise NotImplementedError()

    def likelihood_profile(self, model, parameter, values=None, bounds=2, nvalues=11):
        """Compute likelihood profile for a single parameter of the model.

        Parameters
        ----------
        model : `~gammapy.spectrum.models.SpectralModel`
            Model to compute the likelihood profile for.
        parameter : str
            Parameter to calculate profile for
        values : `~astropy.units.Quantity` (optional)
            Parameter values to evaluate the likelihood for.
        bounds : int or tuple of float
            When an `int` is passed the bounds are computed from `bounds * sigma`
            from the best fit value of the parameter, where `sigma` corresponds to
            the one sigma error on the parameter. If a tuple of floats is given
            those are taken as the min and max values and `nvalues` are linearly
            spaced between those.
        nvalues : int
            Number of parameter grid points to use.

        Returns
        -------
        likelihood_profile : dict
            Dict of parameter values and likelihood values.
        """
        self._model = model.copy()

        likelihood = []

        if values is None:
            if isinstance(bounds, tuple):
                parmin, parmax = bounds
            else:
                parerr = model.parameters.error(parameter)
                parval = model.parameters[parameter].value
                parmin, parmax = parval - bounds * parerr, parval + bounds * parerr

            values = np.linspace(parmin, parmax, nvalues)

        for value in values:
            self._model.parameters[parameter].value = value
            stat = self.total_stat(self._model.parameters)
            likelihood.append(stat)

        return {"values": values, "likelihood": np.array(likelihood)}

    # TODO: delete once Axel removes the caller in FluxPointEstimator
    def sqrt_ts(self, parameters):
        """Compute the sqrt(TS) of a model against the null hypthesis, that
        the amplitude of the model is zero.
        """
        stat_best_fit = self.total_stat(parameters)

        # store best fit amplitude, set amplitude of fit model to zero
        amplitude = parameters["amplitude"].value
        parameters["amplitude"].value = 0
        stat_null = self.total_stat(parameters)

        # set amplitude of fit model to best fit amplitude
        parameters["amplitude"].value = amplitude

        # compute sqrt TS
        ts = np.abs(stat_null - stat_best_fit)
        return np.sign(amplitude) * np.sqrt(ts)


class CovarResult(object):
    """Covar result object."""

    def __init__(self, model, success, nfev):
        self._model = model
        self._success = success
        self._nfev = nfev

    @property
    def model(self):
        """Best fit model."""
        return self._model

    @property
    def success(self):
        """Fit success status flag."""
        return self._success

    @property
    def nfev(self):
        """Number of function evaluations."""
        return self._nfev


class FitResult(object):
    """Fit result object."""

    def __init__(self, model, success, nfev, total_stat, message, backend, method):
        self._model = model
        self._success = success
        self._nfev = nfev
        self._total_stat = total_stat
        self._message = message
        self._backend = backend
        self._method = method

    @property
    def model(self):
        """Best fit model."""
        return self._model

    @property
    def success(self):
        """Fit success status flag."""
        return self._success

    @property
    def nfev(self):
        """Number of function evaluations."""
        return self._nfev

    @property
    def total_stat(self):
        """Value of the fit statistic at minimum."""
        return self._total_stat

    @property
    def message(self):
        """Optimizer status message."""
        return self._message

    @property
    def backend(self):
        """Optimizer backend used for the fit."""
        return self._backend

    @property
    def method(self):
        """Optimizer method used for the fit."""
        return self._method

    def __repr__(self):
        str_ = self.__class__.__name__
        str_ += "\n\n"
        str_ += "\tbackend    : {}\n".format(self.backend)
        str_ += "\tmethod     : {}\n".format(self.method)
        str_ += "\tsuccess    : {}\n".format(self.success)
        str_ += "\tnfev       : {}\n".format(self.nfev)
        str_ += "\ttotal stat : {:.2f}\n".format(self.total_stat)
        str_ += "\tmessage    : {}\n".format(self.message)
        return str_
