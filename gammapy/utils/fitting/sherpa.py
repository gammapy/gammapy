# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from .likelihood import Likelihood

__all__ = ["optimize_sherpa", "covariance_sherpa"]


def get_sherpa_optimizer(name):
    from sherpa.optmethods import LevMar, NelderMead, MonCar, GridSearch

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
        return self.function(self.parameters), 0


def optimize_sherpa(parameters, function, **kwargs):
    """Sherpa optimization wrapper method.

    Parameters
    ----------
    parameters : `~gammapy.utils.modeling.Parameters`
        Parameter list with starting values.
    function : callable
        Likelihood function
    **kwargs : dict
        Options passed to the optimizer instance.

    Returns
    -------
    result : (factors, info, optimizer)
        Tuple containing the best fit factors, some info and the optimizer instance.
    """
    method = kwargs.pop("method", "simplex")
    optimizer = get_sherpa_optimizer(method)
    optimizer.config.update(kwargs)

    pars = [par.factor for par in parameters.parameters]
    parmins = [par.factor_min for par in parameters.parameters]
    parmaxes = [par.factor_max for par in parameters.parameters]

    statfunc = SherpaLikelihood(function, parameters)

    with np.errstate(invalid="ignore"):
        result = optimizer.fit(
            statfunc=statfunc.fcn, pars=pars, parmins=parmins, parmaxes=parmaxes
        )

    factors = result[1]
    info = {"success": result[0], "message": result[3], "nfev": result[4]["nfev"]}
    optimizer = optimizer

    return factors, info, optimizer


def covariance_sherpa():
    raise NotImplementedError
