# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Define `astropy.modeling.fitting.Fitter` sub-classes for our fit statistics.
"""
import numpy as np
from astropy.modeling.fitting import Fitter
from .fit_statistics import cash, cstat

__all__ = ['PoissonLikelihoodFitter']


class PoissonLikelihoodFitter(Fitter):
    """Poisson likelihood fitter.

    The available fit statistics are:
    * 'cash' = `~gammapy.stats.cash`
    * 'cstat' = `~gammapy.stats.cstat`

    Calls the `scipy.optimize.minimize` optimization function.

    Parameters
    ----------
    statistic : {'cash', 'ctstat'}
        Fit statistic
    """
    DEFAULT_FIT_STATISTIC = 'cash'
    FIT_STATISTICS = dict(cash=cash, cstat=cstat)
    supported_constraints = ['fixed']

    def __init__(self, statistic='cash'):

        from scipy.optimize import minimize
        optimizer = minimize

        try:
            statistic = self.FIT_STATISTICS[statistic]
        except KeyError:
            raise ValueError('Invalid statistic: {0}'.format(statistic))

        super(PoissonLikelihoodFitter, self).__init__(optimizer, statistic)


#     def objective_function(self, fitparams, *args):
#         """The Cash Poisson likelihood fit statistic.
# 
#         Parameters
#         ----------
#         fitparams : `numpy.array`
#             Array of fit parameters
#         args : (model, x, y, dx, stat)
#             Tuple with auxiliary 
# 
#         Returns
#         -------
#         stat : float
#             Poisson likelihood fit statistic
#         """
#         model, x, y, dx, stat = args
#         self._fitter_to_model_params(model, fitparams)
#         y_model = dx * model(x)
#         return stat(y, y_model).sum()

    def __call__(self, model, x, y, dx=None):
        """Execute the likelihood minimization.

        Parameters
        ----------
        x : array_like
            x-coordinate
        y : array_like
            Observed number of counts at ``x``
        dx : array_like
            x-bin width
        """

        if dx is None:
            dx = np.ones_like(x)

        import astropy.modeling.fitting as amf

        model_copy = amf._validate_model(model, self._opt_method.supported_constraints)
        farg = amf._convert_input(x, y, z)
        farg = (model_copy, ) + farg
        p0, _ = amf._model_to_fit_params(model_copy)
        fitparams, self.fit_info = self._opt_method(
            self.objective_function, p0, farg)
        amf._fitter_to_model_params(model_copy, fitparams)

        return model_copy
        self._validate_constraints(model)

        model_copy = model.copy()
        x0, _ = self._model_to_fit_params(model_copy)
        result = self.optimizer(self.objective_function, x0=x0,
                                args=(model_copy, x, y, dx))
        self._fitter_to_model_params(model_copy, result.x)

        return model_copy
