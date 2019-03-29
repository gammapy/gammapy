# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from scipy.optimize import minimize
from .likelihood import Likelihood


__all__ = ["optimize_scipy", "covariance_scipy"]


def optimize_scipy(parameters, function, **kwargs):
    method = kwargs.pop("method", "Nelder-Mead")
    pars = [par.factor for par in parameters.free_parameters]

    bounds = []
    for par in parameters.free_parameters:
        parmin = par.factor_min if not np.isnan(par.factor_min) else None
        parmax = par.factor_max if not np.isnan(par.factor_max) else None
        bounds.append((parmin, parmax))

    likelihood = Likelihood(function, parameters)
    result = minimize(likelihood.fcn, pars, bounds=bounds, method=method, **kwargs)

    factors = result.x
    info = {"success": result.success, "message": result.message, "nfev": result.nfev}
    optimizer = None

    return factors, info, optimizer


# TODO: implement, e.g. with numdifftools.Hessian
def covariance_scipy(parameters, function):
    raise NotImplementedError
