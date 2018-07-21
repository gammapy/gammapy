# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""iminuit fitting functions.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np


__all__ = [
    'fit_iminuit',
]


def fit_iminuit(parameters, function, opts_minuit=None):
    """iminuit optimization

    The input `~gammapy.utils.modeling.ParameterList` is copied internally
    before the fit and will thus not be modified. The best-fit parameter values
    are contained in the output `~gammapy.utils.modeling.ParameterList` or the
    `~iminuit.Minuit` object.

    Parameters
    ----------
    parameters : `~gammapy.utils.modeling.ParameterList`
        Parameters with starting values
    function : callable
        Likelihood function
    opts_minuit : dict (optional)
        Options passed to `iminuit.Minuit` constructor

    Returns
    -------
    parameters : `~gammapy.utils.modeling.ParameterList`
        Parameters with best-fit values
    minuit : `~iminuit.Minuit`
        Minuit object
    """
    from iminuit import Minuit

    parameters = parameters.copy()
    minuit_func = MinuitFunction(function, parameters)

    if opts_minuit is None:
        opts_minuit = {}
    opts_minuit.update(make_minuit_par_kwargs(parameters))

    minuit = Minuit(minuit_func.fcn,
                    forced_parameters=parameters.names,
                    **opts_minuit)

    minuit.migrad()
    return parameters, minuit


class MinuitFunction(object):
    """Wrapper for iminuit

    Parameters
    ----------
    parameters : `~gammapy.utils.modeling.ParameterList`
        Parameters with starting values
    function : callable
        Likelihood function
    """

    def __init__(self, function, parameters):
        self.function = function
        self.parameters = parameters

    def fcn(self, *p):
        for parval, par in zip(p, self.parameters.parameters):
            par.value = parval
        val = self.function(self.parameters)
        return val


def make_minuit_par_kwargs(parameters):
    """Create *Parameter Keyword Arguments* for the `Minuit` constructor.

    See: http://iminuit.readthedocs.io/en/latest/api.html#iminuit.Minuit
    """
    kwargs = dict()
    for par in parameters.parameters:
        kwargs[par.name] = par.value
        if par.frozen:
            kwargs['fix_{}'.format(par.name)] = True
        limits = par.parmin, par.parmax
        limits = np.where(np.isnan(limits), None, limits)
        kwargs['limit_{}'.format(par.name)] = limits

        if parameters.covariance is not None:
            err = parameters.error(par.name)
            if err != '0':
                kwargs['error_{}'.format(par.name)] = err

        # TODO: Check if we need 0.5 or 1
        kwargs['errordef'] = 1

    return kwargs
