# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""iminuit fitting functions."""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
from .likelihood import Likelihood

__all__ = ["optimize_iminuit", "covariance_iminuit", "confidence_iminuit", "mncontour"]

log = logging.getLogger(__name__)


class MinuitLikelihood(Likelihood):
    """Likelihood function interface for iminuit."""

    def fcn(self, *factors):
        self.parameters.set_parameter_factors(factors)
        return self.function(self.parameters)


def optimize_iminuit(parameters, function, **kwargs):
    """iminuit optimization

    Parameters
    ----------
    parameters : `~gammapy.utils.modeling.Parameters`
        Parameters with starting values
    function : callable
        Likelihood function
    **kwargs : dict
        Options passed to `iminuit.Minuit` constructor

    Returns
    -------
    result : (factors, info, optimizer)
        Tuple containing the best fit factors, some info and the optimizer instance.
    """
    from iminuit import Minuit

    # In Gammapy, we have the factor 2 in the likelihood function
    # This means `errordef=1` in the Minuit interface is correct
    kwargs.setdefault("errordef", 1)
    kwargs.setdefault("print_level", 0)
    kwargs.update(make_minuit_par_kwargs(parameters))

    minuit_func = MinuitLikelihood(function, parameters)

    minuit = Minuit(minuit_func.fcn, **kwargs)
    minuit.migrad()

    factors = minuit.args
    info = {
        "success": minuit.migrad_ok(),
        "nfev": minuit.get_num_call_fcn(),
        "message": _get_message(minuit),
    }
    optimizer = minuit
    return factors, info, optimizer


def covariance_iminuit(minuit):
    # TODO: add minuit.hesse() call once we have better tests

    message, success = "Hesse terminated successfully.", True
    try:
        covariance_factors = _get_covariance(minuit)
    except TypeError:
        N = len(minuit.args)
        covariance_factors = np.nan * np.ones((N, N))
        message, success = "Hesse failed", False
    return covariance_factors, {"success": success, "message": message}


def confidence_iminuit(minuit, parameters, parameter, sigma, maxcall=0):
    # TODO: this is ugly - design something better for translating to MINUIT parameter names.
    # Maybe a wrapper class MinuitParameters?
    idx = parameters._get_idx(parameter)
    var = _make_parname(idx, parameters[idx])

    message, success = "Minos terminated successfully.", True
    try:
        result = minuit.minos(var=var, sigma=sigma, maxcall=maxcall)
        info = result[var]
    except RuntimeError as error:
        message, success = str(error), False
        info = {"is_valid": False, "lower": np.nan, "upper": np.nan, "nfcn": 0}

    return {
        "success": success,
        "message": message,
        "errp": info["upper"],
        "errn": -info["lower"],
        "nfev": info["nfcn"],
    }


def mncontour(minuit, parameters, x, y, numpoints, sigma):
    idx = parameters._get_idx(x)
    x = _make_parname(idx, parameters[idx])

    idx = parameters._get_idx(y)
    y = _make_parname(idx, parameters[idx])

    x_info, y_info, contour = minuit.mncontour(x, y, numpoints, sigma)
    contour = np.array(contour)

    success = x_info["is_valid"] and y_info["is_valid"]

    return {
        "success": success,
        "x": contour[:, 0],
        "y": contour[:, 1],
        "x_info": x_info,
        "y_info": y_info,
    }


# this code is copied from https://github.com/iminuit/iminuit/blob/master/iminuit/_minimize.py#L95
def _get_message(m):
    message = "Optimization terminated successfully."
    success = m.migrad_ok()
    if not success:
        message = "Optimization failed."
        fmin = m.get_fmin()
        if fmin.has_reached_call_limit:
            message += " Call limit was reached."
        if fmin.is_above_max_edm:
            message += " Estimated distance to minimum too large."
    return message


def _make_parnames(parameters):
    return [_make_parname(idx, par) for idx, par in enumerate(parameters)]


def _make_parname(idx, par):
    return "par_{:03d}_{}".format(idx, par.name)


def make_minuit_par_kwargs(parameters):
    """Create *Parameter Keyword Arguments* for the `Minuit` constructor.

    See: http://iminuit.readthedocs.io/en/latest/api.html#iminuit.Minuit
    """
    names = _make_parnames(parameters)
    kwargs = {"forced_parameters": names}

    for name, par in zip(names, parameters):
        kwargs[name] = par.factor

        min_ = None if np.isnan(par.factor_min) else par.factor_min
        max_ = None if np.isnan(par.factor_max) else par.factor_max
        kwargs["limit_{}".format(name)] = (min_, max_)

        kwargs["error_{}".format(name)] = 1
        kwargs["fix_{}".format(name)] = par.frozen

    return kwargs


def _get_covariance(minuit):
    """Get full covariance matrix as Numpy array.

    This was added as `minuit.np_covariance` in `iminuit` in v1.3,
    but we still want to support v1.2
    """
    n = len(minuit.parameters)
    m = np.zeros((n, n))
    for i1, k1 in enumerate(minuit.parameters):
        for i2, k2 in enumerate(minuit.parameters):
            if {k1, k2} <= set(minuit.list_of_vary_param()):
                m[i1, i2] = minuit.covariance[(k1, k2)]
    return m
