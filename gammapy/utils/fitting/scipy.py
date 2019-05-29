# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from scipy.optimize import minimize, brentq
from scipy.optimize.zeros import RootResults
from .likelihood import Likelihood
from ..interpolation import interpolate_likelihood_profile

__all__ = [
    "optimize_scipy",
    "covariance_scipy",
    "confidence_scipy",
    "likelihood_profile_ul_scipy",
]


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


class TSDifference:
    """Likelihood wrapper to compute TS differences"""

    def __init__(self, function, parameters, parameter, reoptimize, ts_diff):
        self.loglike_ref = function()
        self.parameters = parameters
        self.function = function
        self.parameter = parameter
        self.parameter.frozen = True
        self.ts_diff = ts_diff
        self.reoptimize = reoptimize

    def fcn(self, factor):
        self.parameter.factor = factor
        if self.reoptimize:
            optimize_scipy(self.parameters, self.function, method="L-BFGS-B")
        value = self.function() - self.loglike_ref - self.ts_diff
        return value


def _confidence_scipy_brentq(
    parameters, parameter, function, sigma, reoptimize, upper=True, **kwargs
):
    ts_diff = TSDifference(
        function, parameters, parameter, reoptimize, ts_diff=sigma ** 2
    )

    kwargs.setdefault("a", parameter.factor)

    bound = parameter.factor_max if upper else parameter.factor_min

    if np.isnan(bound):
        bound = parameter.factor
        if upper:
            bound += 1e2 * parameters.error(parameter) / parameter.scale
        else:
            bound -= 1e2 * parameters.error(parameter) / parameter.scale

    kwargs.setdefault("b", bound)

    message, success = "Confidence terminated successfully.", True

    try:
        result = brentq(ts_diff.fcn, full_output=True, **kwargs)
    except ValueError:
        message = (
            "Confidence estimation failed, because bracketing interval"
            " does not contain a unique solution. Try setting the interval by hand."
        )
        success = False
        result = (
            np.nan,
            RootResults(root=np.nan, iterations=0, function_calls=0, flag=0),
        )

    suffix = "errp" if upper else "errn"

    return {
        "nfev_" + suffix: result[1].iterations,
        suffix: np.abs(result[0] - kwargs["a"]),
        "success_" + suffix: success,
        "message_" + suffix: message,
        "loglike_ref": ts_diff.loglike_ref,
    }


def confidence_scipy(parameters, parameter, function, sigma, reoptimize=True, **kwargs):

    if len(parameters.free_parameters) <= 1:
        reoptimize = False

    with parameters.restore_values:
        result = _confidence_scipy_brentq(
            parameters=parameters,
            parameter=parameter,
            function=function,
            sigma=sigma,
            upper=False,
            reoptimize=reoptimize,
            **kwargs
        )

    with parameters.restore_values:
        result_errp = _confidence_scipy_brentq(
            parameters=parameters,
            parameter=parameter,
            function=function,
            sigma=sigma,
            upper=True,
            reoptimize=reoptimize,
            **kwargs
        )

    result.update(result_errp)
    return result


# TODO: implement, e.g. with numdifftools.Hessian
def covariance_scipy(parameters, function):
    raise NotImplementedError


def likelihood_profile_ul_scipy(
    value_scan, dloglike_scan, delta_ts=4, interp_scale="sqrt", **kwargs
):
    """Compute upper limit of a parameter from a likelihood profile.

    Parameters
    ----------
    value_scan : `~numpy.ndarray`
        Array of parameter values.
    dloglike_scan : `~numpy.ndarray`
        Array of delta log-likelihood values, with respect to the minimum.
    delta_ts : float
        Difference in test statistics for the upper limit.
    interp_scale : {"sqrt", "lin"}
        Interpolation scale applied to the likelihood profile. If the profile is
        of parabolic shape, a "sqrt" scaling is recommended. In other cases or
        for fine sampled profiles a "lin" can also be used.
    **kwargs : dict
        Keyword arguments passed to `~scipy.optimize.brentq`.

    Returns
    -------
    ul : float
        Upper limit value.
    """
    interp = interpolate_likelihood_profile(
        value_scan, dloglike_scan, interp_scale=interp_scale
    )

    def f(x):
        return interp((x,)) - delta_ts

    idx = np.argmin(dloglike_scan)
    norm_best_fit = value_scan[idx]
    ul = brentq(f, a=norm_best_fit, b=value_scan[-1], **kwargs)

    return ul
