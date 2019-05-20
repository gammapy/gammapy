# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from scipy.optimize import minimize, brentq
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


def confidence_scipy(parameters, parameter, function, sigma, **kwargs):

    parameter.frozen = True
    loglike = function()

    def f(factor):
        parameter.factor = factor
        _ = optimize_scipy(parameters, function)
        value = (function() - loglike) - sigma ** 2
        return value

    kwargs.setdefault("a", parameter.factor)

    if np.isnan(parameter.factor_max):
        b = parameter.factor + 1e2 * parameters.error(parameter) / parameter.scale
    else:
        b = parameter.factor_max

    kwargs.setdefault("b", b)
    kwargs.setdefault("rtol", 0.01)

    try:
        result = brentq(f, full_output=True, **kwargs)
    except ValueError:
        pass

    message, success = "Scipy confidence terminated successfully.", True

    return {
        "success": success,
        "message": message,
        "errp": result[0],
        "errn": 0,
        "nfev": result[1].iterations,
    }


# TODO: implement, e.g. with numdifftools.Hessian
def covariance_scipy(parameters, function):
    raise NotImplementedError
