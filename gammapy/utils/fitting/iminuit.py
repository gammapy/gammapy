# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""iminuit fitting functions.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np

__all__ = ["optimize_iminuit"]

log = logging.getLogger(__name__)


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
    result : (factors, info, optmizer)
        Tuple containing the best fit factors, some info and the optimizer instance.
    """
    from iminuit import Minuit

    # In Gammapy, we have the factor 2 in the likelihood function
    # This means `errordef=1` in the Minuit interface is correct
    kwargs.setdefault("errordef", 1)
    kwargs.setdefault("print_level", 0)
    kwargs.update(make_minuit_par_kwargs(parameters))

    parnames = _make_parnames(parameters)
    minuit_func = MinuitFunction(function, parameters)

    minuit = Minuit(minuit_func.fcn, forced_parameters=parnames, **kwargs)
    minuit.migrad()

    info = {
        "success": minuit.migrad_ok(),
        "nfev": minuit.get_num_call_fcn(),
        "message": _get_message(minuit),
    }
    return minuit.args, info, minuit


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
    """Create list with unambigious parameter names"""
    return [
        "par_{:03d}_{}".format(idx, par.name)
        for idx, par in enumerate(parameters.parameters)
    ]


class MinuitFunction(object):
    """Wrapper for iminuit

    Parameters
    ----------
    parameters : `~gammapy.utils.modeling.Parameters`
        Parameters with starting values
    function : callable
        Likelihood function
    """

    def __init__(self, function, parameters):
        self.function = function
        self.parameters = parameters

    def fcn(self, *factors):
        self.parameters.set_parameter_factors(factors)
        return self.function(self.parameters)


def make_minuit_par_kwargs(parameters):
    """Create *Parameter Keyword Arguments* for the `Minuit` constructor.

    See: http://iminuit.readthedocs.io/en/latest/api.html#iminuit.Minuit
    """
    kwargs = {}
    parnames = _make_parnames(parameters)
    for idx, parname_ in enumerate(parnames):
        par = parameters[idx]
        kwargs[parname_] = par.factor

        min_ = None if np.isnan(par.min) else par.min
        max_ = None if np.isnan(par.max) else par.max
        kwargs["limit_{}".format(parname_)] = (min_, max_)

        kwargs["error_{}".format(parname_)] = 1

        if par.frozen:
            kwargs["fix_{}".format(parname_)] = True

    return kwargs


def _get_covar(minuit):
    """Get full covar matrix as Numpy array.

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
