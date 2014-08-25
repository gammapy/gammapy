# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Background models.
"""
from __future__ import print_function, division
import numpy as np
from astropy.modeling.models import Gaussian1D

__all__ = ['GaussianBand2D']

DEFAULT_SPLINE_KWARGS = dict(k=1, s=0)


class GaussianBand2D(object):
    """Gaussian band model.

    This 2-dimensional model is Gaussian in ``y`` for a given ``x``,
    and the Gaussian parameters can vary in ``x``.

    One application of this model is the diffuse emission along the
    Galactic plane, i.e. ``x = GLON`` and ``y = GLAT``.

    Parameters
    ----------
    table : `~astropy.table.Table`
        Table of Gaussian parameters.
        ``x``, ``amplitude``, ``mean``, ``stddev``.
    spline_kwargs : dict
        Keyword arguments passed to `~scipy.interpolate.UnivariateSpline`
    """

    def __init__(self, table, spline_kwargs=DEFAULT_SPLINE_KWARGS):
        self.table = table
        self.parnames = ['amplitude', 'mean', 'stddev']

        from scipy.interpolate import UnivariateSpline
        s = dict()
        for parname in self.parnames:
            x = self.table['x']
            y = self.table[parname]
            s[parname] = UnivariateSpline(x, y, **spline_kwargs)
        self._par_model = s

    def _eval_y(self, y, pars):
        """Evaluate Gaussian model at a given ``y`` position.
        """
        return Gaussian1D.eval(y, **pars)

    def parvals(self, x):
        """Interpolated parameter values at a given ``x``.
        """
        x = np.asanyarray(x, dtype=float)
        parvals = dict()
        for parname in self.parnames:
            par_model = self._par_model[parname]
            shape = x.shape
            parvals[parname] = par_model(x.flat).reshape(shape)

        return parvals

    def y_model(self, x):
        """Create model at a given ``x`` position.
        """
        x = np.asanyarray(x, dtype=float)
        parvals = self.parvals(x)
        return Gaussian1D(**parvals)

    def eval(self, x, y):
        """Evaluate model at a given position ``(x, y)`` position.
        """
        x = np.asanyarray(x, dtype=float)
        y = np.asanyarray(y, dtype=float)
        parvals = self.parvals(x)
        return self._eval_y(y, parvals)
