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


def optimize_sherpa(parameters, function, optimizer="simplex"):
    """Sherpa optimization wrapper method.

    Parameters
    ----------
    parameters : `~gammapy.utils.modeling.Parameters`
        Parameter list with starting values.
    function : callable
        Likelihood function
    optimizer : {'levmar', 'simplex', 'moncar', 'gridsearch'}
        Which optimizer to use for the fit. See
        http://cxc.cfa.harvard.edu/sherpa/methods/index.html
        for details on the different options available.

    Returns
    -------
    result : dict
        Result dict with the best fit parameters and optimizer info.
    """
    optimizer = get_sherpa_optimiser(optimizer)

    pars = [par.value for par in parameters.parameters]
    parmins = [par.min for par in parameters.parameters]
    parmaxes = [par.max for par in parameters.parameters]

    statfunc = SherpaFunction(function, parameters)

    with np.errstate(invalid='ignore'):
        result = optimizer.fit(
            statfunc=statfunc.fcn, pars=pars, parmins=parmins, parmaxes=parmaxes
        )

    return {
        "success": result[0],
        "factors": result[1],
        "message": result[3],
        "nfev": result[4]["nfev"],
    }
