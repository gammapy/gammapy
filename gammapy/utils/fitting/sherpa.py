from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = [
    'fit_sherpa',
]


def get_sherpa_optimiser(name):
    from sherpa.optmethods import LevMar, NelderMead, MonCar, GridSearch
    return {
        'levmar': LevMar,
        'simplex': NelderMead,
        'moncar': MonCar,
        'gridsearch': GridSearch,
    }[name]()


class SherpaFunction(object):
    """Wrapper for Sherpa

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

    def fcn(self, factors):
        self.parameters.optimiser_set_factors(factors)
        return self.function(self.parameters)


def fit_sherpa(parameters, function, optimizer='simplex'):
    """Sherpa optimization wrapper method.

    Parameters
    ----------
    parameters : `~gammapy.utils.modeling.ParameterList`
        Parameter list with starting values.
    function : callable
        Likelihood function
    optimizer : {'levmar', 'simplex', 'moncar', 'gridsearch'}
        Which optimizer to use for the fit. See
        http://cxc.cfa.harvard.edu/sherpa/methods/index.html
        for details on the different options available.

    Returns
    -------
    parameters : `~gammapy.utils.modeling.ParameterList`
        Parameter list with best-fit values
    """
    optimizer = get_sherpa_optimiser(optimizer)
    # parameters.optimiser_rescale_parameters()

    pars = [par.value for par in parameters.parameters]
    parmins = [par.min for par in parameters.parameters]
    parmaxes = [par.max for par in parameters.parameters]

    statfunc = SherpaFunction(function, parameters)

    result = optimizer.fit(
        statfunc=statfunc.fcn,
        pars=pars,
        parmins=parmins,
        parmaxes=parmaxes,
    )

    result = {
        'success': result[0],
        'factors': result[1],
        'statval': result[2],
        'message': result[3],
        'info': result[4],  # that's a dict, content varies based on optimiser
    }

    result['nfev'] = result['info']['nfev']

    # Copy final results into the parameters object
    parameters.optimiser_set_factors(result['factors'])

    return result
