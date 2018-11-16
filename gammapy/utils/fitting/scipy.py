# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from .likelihood import Likelihood

__all__ = ["optimize_scipy", "covariance_scipy"]


def optimize_scipy(parameters, function, **kwargs):
    from scipy.optimize import minimize

    pars = [par.factor for par in parameters.parameters]
    likelihood = Likelihood(function, parameters)

    # TODO: understand options for this optimiser
    tol = kwargs.pop("tol", 1e-2)
    result = minimize(likelihood.fcn, pars, tol=tol, **kwargs)

    factors = result.x
    info = {"success": result.success, "message": result.message, "nfev": result.nfev}
    optimizer = None

    return factors, info, optimizer


# TODO: implement, e.g. with numdifftools.Hessian
def covariance_scipy(parameters, function):
    raise NotImplementedError
