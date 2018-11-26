# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import abc
import numpy as np
from astropy.utils.misc import InheritDocstrings
from ...extern import six
from .iminuit import optimize_iminuit, covariance_iminuit, confidence_iminuit, mncontour
from .sherpa import optimize_sherpa, covariance_sherpa
from .scipy import optimize_scipy, covariance_scipy

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
        "covariance": {
            "minuit": covariance_iminuit,
            "sherpa": covariance_sherpa,
            "scipy": covariance_scipy,
        },
        "confidence": {
            "minuit": confidence_iminuit,
            # "sherpa": confidence_sherpa,
            # "scipy": confidence_scipy,
        },
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

    # TODO: probably we should change the `Fit` class to be coupled
    # to a likelihood object, not a model object
    # To facilitate this evolution, we centralise the coupling
    # in this property in a single place,
    # and only use `parameters` from `Fit`, not `model`.
    @property
    def _parameters(self):
        return self._model.parameters

    def run(self, optimize_opts=None, covariance_opts=None):
        """
        Run all fitting steps.

        Parameters
        ----------
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
        optimize_result = self.optimize(**optimize_opts)

        if covariance_opts is None:
            covariance_opts = {}

        covariance_opts.setdefault("backend", "minuit")

        if covariance_opts["backend"] not in registry.register["covariance"]:
            log.warning("No covariance estimate - not supported by this backend.")
            return optimize_result

        covariance_result = self.covariance(**covariance_opts)
        # TODO: not sure how best to report the results
        # back or how to form the FitResult object.
        optimize_result._model = covariance_result.model
        optimize_result._success = optimize_result.success and covariance_result.success
        optimize_result._nfev += covariance_result.nfev

        return optimize_result

    def optimize(self, backend="minuit", **kwargs):
        """Run the optimization.

        Parameters
        ----------
        backend : str
            Which backend to use (see ``gammapy.utils.fitting.registry``)
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
        parameters = self._parameters

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
            model=self._model,
            total_stat=self.total_stat(self._model.parameters),
            backend=backend,
            method=kwargs.get("method", backend),
            **info
        )

    def covariance(self, backend="minuit"):
        """Estimate the covariance matrix.

        Assumes that the model parameters are already optimised.

        Parameters
        ----------
        backend : str
            Which backend to use (see ``gammapy.utils.fitting.registry``)

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
                if hasattr(self, "minuit"):
                    covariance_factors, info = compute(self.minuit)
                else:
                    raise RuntimeError("To use minuit, you must first optimize.")
            else:
                function = self.total_stat
                covariance_factors, info = compute(parameters, function)

        parameters.set_covariance_factors(covariance_factors)

        # TODO: decide what to return, and fill the info correctly!
        return CovarianceResult(model=self._model, success=info["success"], nfev=0)

    def confidence(self, parameter, backend="minuit", sigma=1, **kwargs):
        """Estimate confidence interval.

        Extra ``kwargs`` are passed to the backend.
        E.g. `iminuit.Minuit.minos` supports a ``maxcall`` option.

        Parameters
        ----------
        backend : str
            Which backend to use (see ``gammapy.utils.fitting.registry``)
        parameter : `~gammapy.utils.fitting.Parameter`
            Parameter of interest
        sigma : float
            Number of standard deviations for the confidence level

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
                raise NotImplementedError()

        errp = parameter.scale * result["errp"]
        errn = parameter.scale * result["errn"]

        return {
            "errp": errp,
            "errn": errn,
            "success": result["success"],
            "nfev": result["nfev"],
        }

    def likelihood_profile(self, parameter, values=None, bounds=2, nvalues=11):
        """Compute likelihood profile.

        The method used is to vary one parameter, keeping all others fixed.
        So this is taking a "slice" or "scan" of the likelihood.

        See also: `Fit.minos_profile`.

        Parameters
        ----------
        parameter : `~gammapy.utils.fitting.Parameter`
            Parameter of interest
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
        results : dict
            Dictionary with keys "values" and "likelihood".
        """
        parameters = self._parameters
        parameter = parameters[parameter]

        if values is None:
            if isinstance(bounds, tuple):
                parmin, parmax = bounds
            else:
                parerr = parameters.error(parameter)
                parval = parameter.value
                parmin, parmax = parval - bounds * parerr, parval + bounds * parerr

            values = np.linspace(parmin, parmax, nvalues)

        likelihood = []
        with parameters.restore_values:
            for value in values:
                parameter.value = value
                stat = self.total_stat(parameters)
                likelihood.append(stat)

        return {"values": values, "likelihood": np.array(likelihood)}

    def minos_profile(self):
        """Compute MINOS profile.

        The method used is to vary one parameter,
        then re-optimise all other free parameters
        and to take the likelihood at that point.

        See also: `Fit.likelihood_profile`

        Calls ``iminuit.Minuit.mnprofile``
        """
        raise NotImplementedError

    def likelihood_contour(self):
        """Compute likelihood contour.

        The method used is to vary two parameters, keeping all others fixed.
        So this is taking a "slice" or "scan" of the likelihood.

        See also: `Fit.minos_contour`

        Parameters
        ----------
        TODO

        Returns
        -------
        TODO
        """
        raise NotImplementedError

    def minos_contour(self, x, y, numpoints=10, sigma=1.0):
        """Compute MINOS contour.

        Calls ``iminuit.Minuit.mncontour``.

        This is a contouring algorithm for a 2D function
        which is not simply the likelihood function.
        That 2D function is given at each point ``(par_1, par_2)``
        by re-optimising all other free parameters,
        and taking the likelihood at that point.

        Very compute-intensive and slow.

        Parameters
        ----------
        x, y : `~gammapy.utils.fitting.Parameter`
            Parameters of interest
        numpoints : int
            Number of contour points
        sigma : float
            Number of standard deviations for the confidence level

        Returns
        -------
        result : dict
            Dictionary with keys "x", "y" (Numpy arrays with contour points)
            and a boolean flag "success".
            The result objects from ``mncontour`` are in the additional
            keys "x_info" and "y_info".
        """
        parameters = self._parameters
        x = parameters[x]
        y = parameters[y]

        with parameters.restore_values:
            result = mncontour(self.minuit, parameters, x, y, numpoints, sigma)

        x = result["x"] * x.scale
        y = result["y"] * y.scale

        return {
            "x": x,
            "y": y,
            "success": result["success"],
            "x_info": result["x_info"],
            "y_info": result["y_info"],
        }


class CovarianceResult(object):
    """Covariance result object."""

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
