# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""iminuit fitting functions.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from ..modeling import ParameterList, Parameter


__all__ = [
    'fit_minuit',
]


def fit_minuit(parameters, function):
    """iminuit optimization

    Parameters
    ----------
    parameters : `~gammapy.utils.modelling.ParameterList`
        Parameters with starting values
    function : callable
        Likelihood function
    """
    from iminuit import Minuit

    minuit_func = MinuitFunction(function, parameters)
    minuit_kwargs = make_minuit_kwargs(parameters)

    m = Minuit(minuit_func.fcn,
               forced_parameters=parameters.names,
               **minuit_kwargs)

    m.migrad()
    return parameters


class MinuitFunction(object):
    """Wrapper for iminuit

    Parameters
    ----------
    parameters : `~gammapy.utils.modelling.ParameterList`
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


def make_minuit_kwargs(parameters):
    """Create kwargs for iminuit"""
    kwargs = dict()
    for par in parameters.parameters:
        kwargs[par.name] = par.value

    return kwargs
