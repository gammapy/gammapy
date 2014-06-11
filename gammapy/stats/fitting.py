# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Define `astropy.modeling.fitting.Fitter` sub-classes for our fit statistics.
"""
import numpy as np
from astropy.modeling.fitting import Fitter
from .fit_statistics import cash, cstat

__all__ = ['PoissonLikelihoodFitter']


class PoissonLikelihoodFitter(Fitter):
    """Poisson likelihood fitter."""
    FIT_STATISTICS = dict(cash=cash, cstat=cstat)

    def errorfunc(self, fitparams, *args):
        """The Cash Poisson likelihood fit statistic.

        Parameters
        ----------
        fitparams : `numpy.array`
            Array of fit parameters
        args : (model, x, y, dx, stat)
            Tuple with auxiliary 

        Returns
        -------
        stat : float
            Poisson likelihood fit statistic
        """
        model, x, y, dx, stat = args
        self._fitter_to_model_params(model, fitparams)
        y_model = dx * model(x)
        return stat(y, y_model).sum()

    def __call__(self, model, x, y, dx=None, fit_statistic='cash'):
        """Execute the likelihood minimization.

        The available fit statistics are:
        * 'cash' = `~gammapy.stats.cash`
        * 'cstat' = `~gammapy.stats.cstat`

        Calls the `scipy.optimize.minimize` optimization function.

        Parameters
        ----------
        x : array_like
            x-coordinate
        y : array_like
            Observed number of counts at ``x``
        dx : array_like
            x-bin width
        fit_statistic : {'cash', 'cstat'}
            Fit statistic
        """
        from scipy.optimize import minimize

        if dx == None:
            dx = np.ones_like(x)

        try:
            fit_statistic = self.FIT_STATISTICS[fit_statistic]
        except KeyError:
            raise ValueError('Invalid fit statistic: {0}'.format(fit_statistic))

        if not model.fittable:
            raise ValueError("Model must be a subclass of ParametricModel")
        self._validate_constraints(model)

        model_copy = model.copy()
        x0, _ = self._model_to_fit_params(model_copy)
        result = minimize(self.errorfunc, x0=x0,
                          args=(model_copy, x, y, dx, fit_statistic))
        self._fitter_to_model_params(model_copy, result.x)
        return model_copy
