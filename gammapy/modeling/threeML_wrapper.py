# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""3ml minimization wrapper"""
import numpy as np
from astromodels import Parameter as AstroParameter
from .iminuit import MinuitLikelihood


def optimize_3ml(parameters, function, store_trace=False, **kwargs):
    """iminuit optimization.

    Parameters
    ----------
    parameters : `~gammapy.modeling.Parameters`
        Parameters with starting values.
    function : callable
        Likelihood function.
    store_trace : bool, optional
        Store trace of the fit. Default is False.
    **kwargs : dict
        Options passed to `iminuit.Minuit` constructor. If there is an entry
        'migrad_opts', those options will be passed to `iminuit.Minuit.migrad()`.

    Returns
    -------
    result : (factors, info, optimizer)
        Tuple containing the best fit factors, some information and the optimizer instance.
    """
    from threeML.minimizer.minimization import get_minimizer

    method = kwargs.pop("method", "MINUIT")
    optimizer_class = get_minimizer(method)

    func_3ml = MinuitLikelihood(function, parameters, store_trace=store_trace)
    parameters_3ml = {}
    for p in parameters:
        min_ = None if np.isnan(p.factor_min) else p.factor_min
        max_ = None if np.isnan(p.factor_max) else p.factor_max

        parameters_3ml[p.name] = AstroParameter(
            name="".join(filter(str.isalnum, p.name)),
            value=p.factor,
            min_value=min_,
            max_value=max_,
            free=not p.frozen,
        )

    optimizer = optimizer_class(
        parameters=parameters_3ml, function=func_3ml.fcn, **kwargs
    )
    factors, fval = optimizer._minimize()
    info = {"success": None, "message": None, "trace": func_3ml.trace, "nfev": None}

    return factors, info, optimizer
