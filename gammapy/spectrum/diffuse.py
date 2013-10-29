# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np
from astropy.io import fits
from astropy.table import Table
from .utils import EnergyAxis
from . import powerlaw as pl

__all__ = ['GalacticDiffuse']

class GalacticDiffuse(object):
    """Lookup from FITS cube representing diffuse emission.
    Interpolates linearly in log(e), no interpolation in GLON, GLAT.
    
    http://fermi.gsfc.nasa.gov/ssc/data/access/lat/BackgroundModels.html
    http://fermi.gsfc.nasa.gov/ssc/data/analysis/software/aux/gal_2yearp7v6_v0.fits
    """
    def __init__(self, filename=None, interp_kind='linear'):
        from kapteyn.wcs import Projection
        self.filename = filename
        self.interp_kind = interp_kind
        self.data = fits.getdata(self.filename)
        # Note: the energy axis of the FITS cube is unusable.
        # We only use proj for GLON, GLAT and do ENERGY ourselves
        self.proj = Projection(fits.getheader(self.filename))
        self.e_axis = EnergyAxis(Table(self.filename, 'ENERGIES').Energy)

    def __call__(self, glon, glat, e):
        """Linear interpolation in log(e)"""
        return self.flux(glon, glat, e)

    def flux(self, glon, glat, e):
        from scipy.interpolate import interp1d
        self.set_position(glon, glat)
        f = interp1d(self.e_axis.log_e, self.log_f, kind=self.interp_kind)
        return 10 ** f(np.log10(e))
        # return f_from_points(*self.lookup(glon, glat, e))

    def gamma(self, glon, glat, e):
        f = lambda e: self.flux(glon, glat, e)
        return pl.g_from_f(e, f)
        # return g_from_points(*(self.lookup(glon, glat, e)[:-1]))

    def lookup(self, glon, glat, e):
        x, y = self.proj.topixel((glon, glat, 0))[:-1]
        z1, z2, e1, e2 = self.e_axis(e)
        f1, f2 = self.data[z1, y, x], self.data[z2, y, x]
        return [e1, e2, f1, f2, e]

    def set_position(self, glon, glat):
        x, y = self.proj.topixel((glon, glat, 0))[:-1]
        self.log_f = np.log10(self.data[:, y, x])
