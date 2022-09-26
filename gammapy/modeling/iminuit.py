# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""iminuit fitting functions."""
import logging
import numpy as np
from scipy.stats import chi2, norm
from iminuit import Minuit
from .likelihood import Likelihood

__all__ = [
    "confidence_iminuit",
    "contour_iminuit",
    "covariance_iminuit",
    "optimize_iminuit",
]

log = logging.getLogger(__name__)


class MinuitLikelihood(Likelihood):
    """Likelihood function interface for iminuit."""

    def fcn(self, *factors):
        self.parameters.set_parameter_factors(factors)

        total_stat = self.function()

        if self.store_trace:
            self.store_trace_iteration(total_stat)

        return total_stat


def setup_iminuit(parameters, function, store_trace=False, **kwargs):
    minuit_func = MinuitLikelihood(function, parameters, store_trace=store_trace)

    pars, errors, limits = make_minuit_par_kwargs(parameters)

    minuit = Minuit(minuit_func.fcn, name=list(pars.keys()), **pars)
    minuit.tol = kwargs.pop("tol", 0.1)
    minuit.errordef = kwargs.pop("errordef", 1)
    minuit.print_level = kwargs.pop("print_level", 0)
    minuit.strategy = kwargs.pop("strategy", 1)

    for name, error in errors.items():
        minuit.errors[name] = error

    for name, limit in limits.items():
        minuit.limits[name] = limit

    return minuit, minuit_func


def optimize_iminuit(parameters, function, store_trace=False, **kwargs):
    """iminuit optimization

    Parameters
    ----------
    parameters : `~gammapy.modeling.Parameters`
        Parameters with starting values
    function : callable
        Likelihood function
    store_trace : bool
        Store trace of the fit
    **kwargs : dict
        Options passed to `iminuit.Minuit` constructor. If there is an entry
        'migrad_opts', those options will be passed to `iminuit.Minuit.migrad()`.

    Returns
    -------
    result : (factors, info, optimizer)
        Tuple containing the best fit factors, some info and the optimizer instance.
    """
    migrad_opts = kwargs.pop("migrad_opts", {})

    minuit, minuit_func = setup_iminuit(
        parameters=parameters, function=function, store_trace=store_trace, **kwargs
    )

    minuit.migrad(**migrad_opts)

    factors = minuit.values
    info = {
        "success": minuit.valid,
        "nfev": minuit.nfcn,
        "message": _get_message(minuit, parameters),
        "trace": minuit_func.trace,
    }
    optimizer = minuit

    return factors, info, optimizer


def covariance_iminuit(parameters, function, **kwargs):
    minuit = kwargs["minuit"]

    if minuit is None:
        minuit, _ = setup_iminuit(
            parameters=parameters, function=function, store_trace=False, **kwargs
        )
        minuit.hesse()

    message, success = "Hesse terminated successfully.", True

    try:
        covariance_factors = np.array(minuit.covariance)
    except (TypeError, RuntimeError):
        N = len(minuit.values)
        covariance_factors = np.nan * np.ones((N, N))
        message, success = "Hesse failed", False

    return covariance_factors, {"success": success, "message": message}


def confidence_iminuit(parameters, function, parameter, reoptimize, sigma, **kwargs):
    # TODO: this is ugly - design something better for translating to MINUIT parameter names.
    if not reoptimize:
        log.warning("Reoptimize = False ignored for iminuit backend")

    minuit, minuit_func = setup_iminuit(
        parameters=parameters, function=function, store_trace=False, **kwargs
    )
    migrad_opts = kwargs.get("migrad_opts", {})
    minuit.migrad(**migrad_opts)

    # Maybe a wrapper class MinuitParameters?
    parameter = parameters[parameter]
    idx = parameters.free_parameters.index(parameter)
    var = _make_parname(idx, parameter)

    message = "Minos terminated successfully."
    cl = 2 * norm.cdf(sigma) - 1

    try:
        minuit.minos(var, cl=cl, ncall=None)
        info = minuit.merrors[var]
    except (AttributeError, RuntimeError) as error:
        return {
            "success": False,
            "message": str(error),
            "errp": np.nan,
            "errn": np.nan,
            "nfev": 0,
        }

    return {
        "success": info.is_valid,
        "message": message,
        "errp": info.upper,
        "errn": -info.lower,
        "nfev": info.nfcn,
    }


def contour_iminuit(parameters, function, x, y, numpoints, sigma, **kwargs):
    minuit, minuit_func = setup_iminuit(
        parameters=parameters, function=function, store_trace=False, **kwargs
    )
    minuit.migrad()

    par_x = parameters[x]
    idx_x = parameters.free_parameters.index(par_x)
    x = _make_parname(idx_x, par_x)

    par_y = parameters[y]
    idx_y = parameters.free_parameters.index(par_y)
    y = _make_parname(idx_y, par_y)

    cl = chi2(2).cdf(sigma**2)
    contour = minuit.mncontour(x=x, y=y, size=numpoints, cl=cl)
    # TODO: add try and except to get the success
    return {
        "success": True,
        "x": contour[:, 0],
        "y": contour[:, 1],
    }


# this code is copied from https://github.com/iminuit/iminuit/blob/master/iminuit/_minimize.py#L95
def _get_message(m, parameters):
    message = "Optimization terminated successfully."
    success = m.accurate
    success &= np.all(np.isfinite([par.value for par in parameters]))
    if not success:
        message = "Optimization failed."
        fmin = m.fmin
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
    pars, errors, limits = {}, {}, {}

    for name, par in zip(names, parameters.free_parameters):
        pars[name] = par.factor

        min_ = None if np.isnan(par.factor_min) else par.factor_min
        max_ = None if np.isnan(par.factor_max) else par.factor_max
        limits[name] = (min_, max_)

        if par.error == 0 or np.isnan(par.error):
            error = 1
        else:
            error = par.error / par.scale
        errors[name] = error

    return pars, errors, limits
