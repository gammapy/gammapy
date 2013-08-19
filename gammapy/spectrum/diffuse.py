# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy.io import fits
from .utils import EnergyAxis
from . import powerlaw as pl


class GalacticDiffuse:
    """Lookup from FITS cube representing diffuse emission.
    Interpolates linearly in log(e), no interpolation in GLON, GLAT"""
    def __init__(self, filename=None, interp_kind='linear'):
        # TODO: use astropy!
        from kapteyn.wcs import Projection
        from atpy import Table
        if filename != None:
            self.filename = filename
        else:
            self.filename = ('/Users/deil/bin/fermi/ScienceTools-v9r23p1'
                             '-fssc-20110726/external/diffuseModels/'
                             'gal_2yearp7v6_v0.fits')
        self.interp_kind = interp_kind
        print filename
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
        # print glon, glat, e
        x, y = self.proj.topixel((glon, glat, 0))[:-1]
        z1, z2, e1, e2 = self.e_axis(e)
        f1, f2 = self.data[z1, y, x], self.data[z2, y, x]
        # print x, y, z1, z2, e1, e2, f1, f2
        return [e1, e2, f1, f2, e]

    def set_position(self, glon, glat):
        x, y = self.proj.topixel((glon, glat, 0))[:-1]
        self.log_f = np.log10(self.data[:, y, x])

if __name__ == '__main__':
    print GalacticDiffuse()(100, 30, 50)
