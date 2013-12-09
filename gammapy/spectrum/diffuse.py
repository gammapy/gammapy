# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from .utils import EnergyAxis
from . import powerlaw as pl

__all__ = ['GalacticDiffuse']

class GalacticDiffuse(object):
    """Lookup diffuse emission flux from FITS cube as used e.g. by Fermi or GALPROP.

    The lookup is done via interpolation in log(energy),
    no interpolation is done in position coordinates.
    
    Diffuse model files in this format are distributed with the Fermi Science tools
    software and can also be downloaded at
    http://fermi.gsfc.nasa.gov/ssc/data/access/lat/BackgroundModels.html
    
    E.g. the 2-year diffuse model that was used in the 2FGL catalog production is at
    http://fermi.gsfc.nasa.gov/ssc/data/analysis/software/aux/gal_2yearp7v6_v0.fits
    
    Parameters
    ----------
    filename : str
        Filename of the diffuse model file
    interpolation : str
        Type of interpolation method in log(E).
        Passed to `scipy.interpolate.interp1d`
    """
    def __init__(self, filename=None, interpolation='linear'):
        self.filename = filename
        self.interpolation = interpolation
        self.data = fits.getdata(self.filename)
        # Note: the energy axis of the FITS cube is unusable.
        # We only use proj for GLON, GLAT and do ENERGY ourselves
        header = fits.getheader(self.filename)
        self.wcs = WCS(header)
        energy = Table.read(self.filename, 'ENERGIES')['Energy']
        self.e_axis = EnergyAxis(energy)

    def flux(self, glon, glat, energy):
        """Power law differential flux.

        Parameters
        ----------
        glon : array-like
            Galactic longitude (deg)
        glat : array-like
            Galactic latitude (deg)
        energy : array-like
            Energy (MeV)

        Returns
        -------
        flux : array
            Differential flux (cm^-2 s^-1 sr^-1 MeV^-1)
        """
        from scipy.interpolate import interp1d
        self._set_position(glon, glat)
        #import IPython; IPython.embed()
        interpolator = interp1d(self.e_axis.log_e, self.log_f, kind=self.interpolation)
        flux = 10 ** interpolator(np.log10(energy))
        # return f_from_points(*self.lookup(glon, glat, e))
        return flux

    def spectral_index(self, glon, glat, energy):
        """Power law spectral index.
        
        Parameters
        ----------
        glon : array-like
            Galactic longitude (deg)
        glat : array-like
            Galactic latitude (deg)
        energy : array-like
            Energy (MeV)
        
        Returns
        -------
        spectral_index : array
        """
        spectrum = lambda energy_: self.flux(glon, glat, energy_)
        spectral_index = pl.g_from_f(energy, spectrum)
        # return g_from_points(*(self.lookup(glon, glat, e)[:-1]))
        return spectral_index

    def lookup(self, glon, glat, energy):
        """TODO: Unused ... document or remove.
        """
        x, y = self._get_xy(glon, glat)
        z1, z2, energy1, energy2 = self.e_axis(energy)
        f1, f2 = self.data[z1, y, x], self.data[z2, y, x]
        return [energy1, energy2, f1, f2, energy]

    def _set_position(self, glon, glat):
        """Pre-compute log-flux vector for a given position."""
        x, y = self._get_xy(glon, glat)
        self.log_f = np.log10(self.data[:, y, x])
    
    def _get_xy(self, glon, glat):
        """Find pixel coordinates for a given position."""
        # We're not interested in the energy axis, so we give a dummy value of 1
        x, y = self.wcs.wcs_world2pix(glon, glat, 1, 0)[:-1]
        # Find the nearest integer pixel
        x = np.round(x).astype(int)
        y = np.round(y).astype(int)
        return x, y
