# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import logging
import importlib
from .likelihood import Likelihood

log = logging.getLogger(__name__)

__all__ = ["optimize_sherpa", "covariance_sherpa"]


def is_sherpa_available():
    """Check if ray is available."""
    try:
        importlib.import_module("sherpa")
        return True
    except ModuleNotFoundError:
        return False


def get_sherpa_optimizer(name):
    from sherpa.optmethods import GridSearch, LevMar, MonCar, NelderMead

    return {
        "levmar": LevMar,
        "simplex": NelderMead,
        "moncar": MonCar,
        "gridsearch": GridSearch,
    }[name]()


class SherpaLikelihood(Likelihood):
    """Likelihood function interface for Sherpa."""

    def fcn(self, factors):
        self.parameters.set_parameter_factors(factors)
        total_stat = self.function()

        if self.store_trace:
            self.store_trace_iteration(total_stat)

        return total_stat, 0


def optimize_sherpa(parameters, function, store_trace=False, **kwargs):
    """Sherpa optimization wrapper method.

    Parameters
    ----------
    parameters : `~gammapy.modeling.Parameters`
        Parameter list with starting values.
    function : callable
        Likelihood function.
    **kwargs : dict
        Options passed to the optimizer instance.

    Returns
    -------
    result : (factors, info, optimizer)
        Tuple containing the best fit factors, some information and the optimizer instance.
    """
    if is_sherpa_available():
        log.warning("Sherpa is not installed.")
        info = {
            "success": False,
            "message": "Sherpa not installed",
            "nfev": None,
            "trace": None,
        }
        return None, info, None

    method = kwargs.pop("method", "simplex")
    optimizer = get_sherpa_optimizer(method)
    optimizer.config.update(kwargs)

    pars = [par.factor for par in parameters.free_parameters]
    parmins = [par.factor_min for par in parameters.free_parameters]
    parmaxes = [par.factor_max for par in parameters.free_parameters]

    statfunc = SherpaLikelihood(function, parameters, store_trace)

    with np.errstate(invalid="ignore"):
        result = optimizer.fit(
            statfunc=statfunc.fcn, pars=pars, parmins=parmins, parmaxes=parmaxes
        )

    factors = result[1]
    info = {
        "success": result[0],
        "message": result[3],
        "nfev": result[4]["nfev"],
        "trace": statfunc.trace,
    }

    return factors, info, optimizer


def covariance_sherpa():
    raise NotImplementedError
