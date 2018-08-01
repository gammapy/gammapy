from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
from sherpa.optmethods import LevMar, NelderMead, MonCar, GridSearch

SHERPA_OPTMETHODS = OrderedDict()
SHERPA_OPTMETHODS['levmar'] = LevMar()
SHERPA_OPTMETHODS['simplex'] = NelderMead()
SHERPA_OPTMETHODS['moncar'] = MonCar()
SHERPA_OPTMETHODS['gridsearch'] = GridSearch()


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
    optimizer = SHERPA_OPTMETHODS[optimizer]

    pars = [par.value for par in parameters.parameters]
    parmins = [par.min for par in parameters.parameters]
    parmaxes = [par.max for par in parameters.parameters]

    def statfunc(values):
        parameters.update_values_from_tuple(values)
        return function(parameters)

    result = optimizer.fit(
        statfunc=statfunc,
        pars=pars,
        parmins=parmins,
        parmaxes=parmaxes
    )

    pars_best_fit = result[1]

    for par, value in zip(parameters, pars_best_fit):
        par.value = value

    return result