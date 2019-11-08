# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""iminuit fitting functions."""
import logging
import numpy as np
from .likelihood import Likelihood

__all__ = ["optimize_iminuit", "covariance_iminuit", "confidence_iminuit", "mncontour"]

log = logging.getLogger(__name__)


class MinuitLikelihood(Likelihood):
    """Likelihood function interface for iminuit."""

    def fcn(self, *factors):
        self.parameters.set_parameter_factors(factors)
        return self.function()


def optimize_iminuit(parameters, function, **kwargs):
    """iminuit optimization

    Parameters
    ----------
    parameters : `~gammapy.modeling.Parameters`
        Parameters with starting values
    function : callable
        Likelihood function
    **kwargs : dict
        Options passed to `iminuit.Minuit` constructor. If there is an entry 'migrad_opts', those options
        will be passed to `iminuit.Minuit.migrad()`.

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

    kwargs = kwargs.copy()
    migrad_opts = kwargs.pop("migrad_opts", {})
    minuit = Minuit(minuit_func.fcn, **kwargs)
    minuit.migrad(**migrad_opts)

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
        covariance_factors = minuit.np_covariance()
    except (TypeError, RuntimeError):
        N = len(minuit.args)
        covariance_factors = np.nan * np.ones((N, N))
        message, success = "Hesse failed", False
    return covariance_factors, {"success": success, "message": message}


def confidence_iminuit(minuit, parameters, parameter, sigma, maxcall=0):
    # TODO: this is ugly - design something better for translating to MINUIT parameter names.
    # Maybe a wrapper class MinuitParameters?
    parameter = parameters[parameter]
    idx = parameters.free_parameters._get_idx(parameter)
    var = _make_parname(idx, parameter)

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
    par_x = parameters[x]
    idx_x = parameters.free_parameters._get_idx(par_x)
    x = _make_parname(idx_x, par_x)

    par_y = parameters[y]
    idx_y = parameters.free_parameters._get_idx(par_y)
    y = _make_parname(idx_y, par_y)

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
    return f"par_{idx:03d}_{par.name}"


def make_minuit_par_kwargs(parameters):
    """Create *Parameter Keyword Arguments* for the `Minuit` constructor.

    See: http://iminuit.readthedocs.io/en/latest/api.html#iminuit.Minuit
    """
    names = _make_parnames(parameters.free_parameters)
    kwargs = {"forced_parameters": names}

    for name, par in zip(names, parameters.free_parameters):
        kwargs[name] = par.factor

        min_ = None if np.isnan(par.factor_min) else par.factor_min
        max_ = None if np.isnan(par.factor_max) else par.factor_max
        kwargs[f"limit_{name}"] = (min_, max_)

        if parameters.covariance is None:
            error = 1
        else:
            error = parameters.error(par) / par.scale

        kwargs[f"error_{name}"] = error

    return kwargs
