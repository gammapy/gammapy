from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import abc

import numpy as np
from astropy.utils.misc import InheritDocstrings

from ...extern import six
from .iminuit import optimize_iminuit, _get_covar
from .sherpa import optimize_sherpa


__all__ = ["Fit"]

log = logging.getLogger(__name__)


class FitMeta(InheritDocstrings, abc.ABCMeta):
    pass


@six.add_metaclass(FitMeta)
class Fit(object):
    """Abstract Fit base class.
    """

    _optimize_funcs = {"minuit": optimize_iminuit, "sherpa": optimize_sherpa}

    @abc.abstractmethod
    def total_stat(self, parameters):
        """Total likelihood given the current model parameters"""
        pass

    def likelihood_profiles(self, model, parameters="all"):
        """Compute likelihood profiles for multiple parameters.

        Parameters
        ----------
        model : `~gammapy.spectrum.models.SpectralModel` or `~gammapy.cube.models.SkyModel`
            Model to compute the likelihood profile for.
        parameters : list of str or "all"
            For which parameters to compute likelihood profiles.
        """
        profiles = {}

        if parameters == "all":
            parameters = [par.name for par in model.paramaters]

        for parname in parameters:
            profiles[parname] = self.likelihood_profile(model, parname)
        return profiles

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

    def optimize(self, backend="minuit", **kwargs):
        """Run the optimization

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
        fit_result : `dict`
            Optimize info dict with the best fit model and additional information.
        """
        parameters = self._model.parameters

        if parameters.apply_autoscale:
            parameters.autoscale()

        optimize = self._optimize_funcs[backend]
        factors, info, optimizer = optimize(
            parameters=parameters, function=self.total_stat, **kwargs
        )

        # As preliminary solution would like to provide a possibility that the user
        # can access the Minuit object, because it features a lot useful functionality
        if backend == "minuit":
            self.minuit = optimizer

        # Copy final results into the parameters object
        parameters.set_parameter_factors(factors)

        return dict(
            model=self._model.copy(),
            total_stat=self.total_stat(self._model.parameters),
            backend=backend,
            method=kwargs.get("method", backend),
            **info
        )

    # TODO: this is a preliminary solution to restore the old behaviour, that's
    # why the method is hidden.
    def _estimate_errors(self, model):
        """Run the error estimation"""
        parameters = model.parameters

        if hasattr(self, "minuit"):
            covar = _get_covar(self.minuit)
            parameters.set_covariance_factors(covar)
            self._model.parameters.set_covariance_factors(covar)
        else:
            log.warning(
                "No covariance matrix found. Error estimation currently"
                " only works with iminuit backend."
            )
            parameters.covariance = None
        return model

    def run(self, steps="all", optimize_opts=None, profile_opts=None):
        """
        Run all fitting steps.

        Parameters
        ----------
        steps : {"all", "optimize", "errors", "profiles"}
            Which fitting steps to run.
        optimize_opts : dict
            Options passed to `Fit.optimize`.
        profile_opts : dict
            Options passed to `Fit.likelihood_profiles`.

        Returns
        -------
        fit_result : `FitResult`
            Fit result object with the best fit model and additional information.
        """
        if steps == "all":
            steps = ["optimize", "errors"]

        if "optimize" in steps:
            if optimize_opts == None:
                optimize_opts = {}
            result = self.optimize(**optimize_opts)

        if "errors" in steps:
            result["model"] = self._estimate_errors(result["model"])

        if "profiles" in steps:
            if profile_opts == None:
                profile_opts = {}
            result["likelihood_profiles"] = self.likelihood_profiles(
                result["model"], **profile_opts
            )

        return FitResult(**result)


class FitResult(object):
    """Fit result object."""

    def __init__(
        self,
        model,
        success,
        nfev,
        total_stat,
        message,
        backend,
        method,
        likelihood_profiles=None,
    ):
        self._model = model
        self._success = success
        self._nfev = nfev
        self._total_stat = total_stat
        self._message = message
        self._backend = backend
        self._method = method
        self._likelihood_profiles = likelihood_profiles

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
        """Number of function evaluations until convergence or stop."""
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

    @property
    def likelihood_profiles(self):
        """Likelihood profiles for paramaters."""
        return self._likelihood_profiles

    def plot_likelihood_profile(self, parameter, ax=None, **kwargs):
        """Plot likelihood profile for a given parameter.

        Parameters
        ----------
        parameter : str
            Parameter to plot profile for.

        Returns
        -------
        ax : `matplotlib.pyplot.Axes`
            Axes object.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        ts_diff = self.likelihood_profiles[parameter]["likelihood"] - self.total_stat
        values = self.likelihood_profiles[parameter]["values"]

        ax.plot(values, ts_diff, **kwargs)
        unit = self.model.parameters[parameter].unit
        ax.set_xlabel(parameter + "[unit]".format(unit=unit))
        ax.set_ylabel("TS difference")
        return ax
