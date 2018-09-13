from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import abc

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
    _optimize_funcs = {
        "minuit": optimize_iminuit,
        "sherpa": optimize_sherpa,
    }

    @abc.abstractmethod
    def total_stat(self, parameters):
        """Total likelihood given the current model parameters"""
        pass

    def optimize(self, backend="minuit", opts=None):
        """Run the optimization

        Parameters
        ----------
        backend : {"minuit", "sherpa"}
            Which optimizer to use. See https://iminuit.readthedocs.io for details
            on the the option `"minuit"`.
            See http://cxc.cfa.harvard.edu/sherpa/methods/index.html for details
            on the other methods.
        opts : dict (optional)
            Options passed to `iminuit.Minuit` constructor

        Returns
        -------
        fit_result : `FitResult`
            Fit result object with the best fit model and additional information.
        """
        parameters = self._model.parameters

        if parameters.apply_autoscale:
            parameters.autoscale()

        optimize = self._optimize_funcs[backend]

        factors, info, optimizer = optimize(
            parameters=parameters,
            function=self.total_stat,
            opts=opts,
        )

        if backend == "minuit":
            self.minuit = optimizer

        # Copy final results into the parameters object
        parameters.set_parameter_factors(factors)

        return FitResult(
            model=self._model.copy(),
            total_stat=self.total_stat(self._model.parameters),
            backend=backend,
            method=opts.get('method'),
            **info
            )

    # TODO: this is a preliminary solution to restore the old behaviour, that's
    # why the method is hidden.
    def _estimate_errors(self, fit_result):
        """Run the error estimation"""
        parameters = fit_result.model.parameters

        if self._minuit.covariance is not None:
            covar = _get_covar(self._minuit)
            parameters.set_covariance_factors(covar)
            self._model.parameters.set_covariance_factors(covar)
        else:
            log.warning("No covariance matrix found")
            parameters.covariance = None
        return fit_result

    def run(self, steps="all", optimizer='minuit', opts_minuit=None):
        """
        Run all fitting steps.

        Parameters
        ----------
        steps : {"all", "optimize", "errors"}
            Which fitting steps to run.
        optimizer : {"minuit", "levmar", "simplex", "moncar", "gridsearch"}
            Which optimizer to use. See `.optimize()` for details.
        opts_minuit : dict (optional)
            Options passed to `iminuit.Minuit` constructor

        Returns
        -------
        fit_result : `FitResult`
            Fit result object with the best fit model and additional information.
        """
        if steps == "all":
            steps = ["optimize", "errors"]

        if "optimize" in steps:
            result = self.optimize(optimizer=optimizer, opts_minuit=opts_minuit)

        if "errors" in steps:
            result = self._estimate_errors(result)

        return result


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
        str_ += "\tmethod    : {}\n".format(self.method)
        str_ += "\tsuccess    : {}\n".format(self.success)
        str_ += "\tnfev       : {}\n".format(self.nfev)
        str_ += "\ttotal stat : {:.2f}\n".format(self.total_stat)
        str_ += "\tmessage    : {}\n".format(self.message)
        return str_
