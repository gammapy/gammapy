# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""iminuit fitting functions.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np

__all__ = [
    'fit_iminuit',
]

log = logging.getLogger(__name__)

def fit_iminuit(parameters, function, opts_minuit=None):
    """iminuit optimization

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

    minuit_func = MinuitFunction(function, parameters)

    if opts_minuit is None:
        opts_minuit = {}
    opts_minuit.update(make_minuit_par_kwargs(parameters))

    # In Gammapy, we have the factor 2 in the likelihood function
    # This means `errordef=1` in the Minuit interface is correct
    opts_minuit.setdefault('errordef', 1)

    minuit = Minuit(minuit_func.fcn,
                    forced_parameters=parameters.names,
                    **opts_minuit)

    minuit.migrad()

    parameters.update_values_from_tuple(minuit.args)
    if minuit.covariance is not None:
        parameters.covariance = _get_covar(minuit)
    else:
        log.warning("No covariance matrix found")
        parameters.covariance = None

    return minuit


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

    def fcn(self, *values):
        self.parameters.update_values_from_tuple(values)
        return self.function(self.parameters)


def make_minuit_par_kwargs(parameters):
    """Create *Parameter Keyword Arguments* for the `Minuit` constructor.

    See: http://iminuit.readthedocs.io/en/latest/api.html#iminuit.Minuit
    """
    kwargs = {}
    for par in parameters.parameters:
        kwargs[par.fullname] = par.value

        min_ = None if np.isnan(par.min) else par.min
        max_ = None if np.isnan(par.max) else par.max
        kwargs['limit_{}'.format(par.fullname)] = (min_, max_)

        if parameters.covariance is None:
            kwargs['error_{}'.format(par.fullname)] = 1
        else:
            kwargs['error_{}'.format(par.fullname)] = parameters.error(par.name)

        if par.frozen:
            kwargs['fix_{}'.format(par.fullname)] = True

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
            if set([k1, k2]).issubset(minuit.list_of_vary_param()):
                m[i1, i2] = minuit.covariance[(k1, k2)]
    return m
