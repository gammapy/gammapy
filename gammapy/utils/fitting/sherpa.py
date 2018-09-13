from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

__all__ = ["optimize_sherpa"]


def get_sherpa_optimiser(name):
    from sherpa.optmethods import LevMar, NelderMead, MonCar, GridSearch

    return {
        "levmar": LevMar,
        "simplex": NelderMead,
        "moncar": MonCar,
        "gridsearch": GridSearch,
    }[name]()


class SherpaFunction(object):
    """Wrapper for Sherpa

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

    def fcn(self, factors):
        self.parameters.set_parameter_factors(factors)
        return self.function(self.parameters), 0


def optimize_sherpa(parameters, function, opts=None):
    """Sherpa optimization wrapper method.

    Parameters
    ----------
    parameters : `~gammapy.utils.modeling.Parameters`
        Parameter list with starting values.
    function : callable
        Likelihood function
    opts : dict
        Options dict passed to the optimizer. Can contain the "method"
        keyword to choose from {'levmar', 'simplex', 'moncar', 'gridsearch'}.
        See http://cxc.cfa.harvard.edu/sherpa/methods/index.html
        for details on the different options available.

    Returns
    -------
    result : (factors, info, optimizer)
        Tuple containing the best fit factors, some info and the optimizer instance.
    """
    if opts is None:
        opts = {"method": "simplex"}

    optimizer = get_sherpa_optimiser(opts["method"])

    pars = [par.value for par in parameters.parameters]
    parmins = [par.min for par in parameters.parameters]
    parmaxes = [par.max for par in parameters.parameters]

    statfunc = SherpaFunction(function, parameters)

    with np.errstate(invalid='ignore'):
        result = optimizer.fit(
            statfunc=statfunc.fcn, pars=pars, parmins=parmins, parmaxes=parmaxes
        )

    info = {
        "success": result[0],
        "message": result[3],
        "nfev": result[4]["nfev"],
    }

    return result[1], info, optimizer
