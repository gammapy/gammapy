# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Background models.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy.modeling.models import Gaussian1D
import astropy.units as u
from astropy.units import Quantity
from astropy.coordinates import Angle
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from ..utils.wcs import (linear_wcs_to_arrays,
                         linear_arrays_to_wcs)
from ..utils.fits import table_to_fits_table

__all__ = ['GaussianBand2D',
           'CubeBackgroundModel',
           ]

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

    def _evaluate_y(self, y, pars):
        """Evaluate Gaussian model at a given ``y`` position.
        """
        return Gaussian1D.evaluate(y, **pars)

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

    def evaluate(self, x, y):
        """Evaluate model at a given position ``(x, y)`` position.
        """
        x = np.asanyarray(x, dtype=float)
        y = np.asanyarray(y, dtype=float)
        parvals = self.parvals(x)
        return self._evaluate_y(y, parvals)


class CubeBackgroundModel(object):

    """Cube background model.

    Container class for cube background model *(X, Y, energy)*.
    *(X, Y)* are detector coordinates (a.k.a. nominal system).
    The class hass methods for reading a model from a fits file,
    write a model to a fits file and plot the models.

    The order of the axes in the background cube is **(E, y, x)**,
    so in order to access the data correctly, the call is
    ``bg_cube_model.background[energy_bin, dety_bin, detx_bin]``.

    TODO: review this doc!!!
    TODO: review this class!!!
    TODO: review high-level doc!!!

    Parameters
    ----------
    detx_bins : `~astropy.coordinates.Angle`
        Spatial bin edges vector (low and high). X coordinate.
    dety_bins : `~astropy.coordinates.Angle`
        Spatial bin edges vector (low and high). Y coordinate.
    energy_bins : `~astropy.units.Quantity`
        Energy bin edges vector (low and high).
    background : `~astropy.units.Quantity`
        Background cube in (energy, X, Y) format.

    Examples
    --------
    Access cube bg model data:

    .. code:: python

        energy_bin = bg_cube_model.find_energy_bin(energy=Quantity(2., 'TeV'))
        det_bin = bg_cube_model.find_det_bin(det=Angle([0., 0.], 'degree'))
        bg_cube_model.background[energy_bin, det_bin[1], det_bin[0]]
    """
