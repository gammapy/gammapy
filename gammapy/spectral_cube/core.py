# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Gamma-ray spectral cube.

TODO: split `GammaSpectralCube` into a base class ``SpectralCube`` and a few sub-classes:

* ``GammaSpectralCube`` to represent functions evaluated at grid points (diffuse model format ... what is there now).
* ``ExposureCube`` should also be supported (same semantics, but different units / methods as ``GammaSpectralCube`` (``gtexpcube`` format)
* ``SpectralCubeHistogram`` to represent model or actual counts in energy bands (``gtbin`` format)
"""
from __future__ import print_function, division
import numpy as np
from astropy.io import fits
import astropy.units as u
from astropy.units import Quantity
from astropy.table import Table
from astropy.wcs import WCS
from ..spectrum import LogEnergyAxis, energy_bounds_equal_log_spacing
from ..spectrum import powerlaw

__all__ = ['GammaSpectralCube']


class GammaSpectralCube(object):
    """Spectral cube for gamma-ray astronomy.

    Can be used e.g. for Fermi or GALPROP diffuse model cubes.

    Note that there is a very nice ``SpectralCube`` implementation here:
    http://spectral-cube.readthedocs.org/en/latest/index.html

    Here is some discussion if / how it could be used: 
    https://github.com/radio-astro-tools/spectral-cube/issues/110

    For now we re-implement what we need here.

    Parameters
    ----------
    data : array_like
        Data array (3-dim)
    wcs : `~astropy.wcs.WCS`
        Word coordinate system transformation
    energy : `~astropy.units.Quantity`
        Energy array

    Notes
    -----
    Diffuse model files in this format are distributed with the Fermi Science tools
    software and can also be downloaded at
    http://fermi.gsfc.nasa.gov/ssc/data/access/lat/BackgroundModels.html

    E.g. the 2-year diffuse model that was used in the 2FGL catalog production is at
    http://fermi.gsfc.nasa.gov/ssc/data/analysis/software/aux/gal_2yearp7v6_v0.fits
    """
    def __init__(self, data, wcs, energy):
        # TODO: check validity of inputs
        self.data = data
        self.wcs = wcs
        
        # TODO: decide whether we want to use an EnergyAxis object or just use the array directly.
        self.energy = energy
        self.energy_axis = LogEnergyAxis(energy)

        # Initialise the interpolator
        # This doesn't do any computations ... I'm not sure if it allocates extra arrays.
        from scipy.interpolate import RegularGridInterpolator
        points = list(map(np.arange, data.shape))
        self._interpolate = RegularGridInterpolator(points, data, fill_value=None)

    @staticmethod
    def read(filename):
        """Read spectral cube from FITS file.

        Parameters
        ----------
        filename : str
            File name

        Returns
        -------
        spectral_cube : `GammaSpectralCube`
            Spectral cube
        """
        data = fits.getdata(filename)
        data = Quantity(data, '1 / (cm2 MeV s sr)')
        # Note: the energy axis of the FITS cube is unusable.
        # We only use proj for LON, LAT and do ENERGY ourselves
        header = fits.getheader(filename)
        wcs = WCS(header)
        energy = Table.read(filename, 'ENERGIES')['Energy']
        energy = Quantity(energy, 'MeV')

        return GammaSpectralCube(data, wcs, energy)

    def flux(self, lon, lat, energy):
        """Differential flux (linear interpolation).

        Parameters
        ----------
        lon : `~astropy.units.Quantity` or `~astropy.units.Angle`
            Longitude
        lat : `~astropy.units.Quantity` or `~astropy.units.Angle`
            Latitude
        energy : `~astropy.units.Quantity`
            Energy

        Returns
        -------
        flux : `~astropy.units.Quantity`
            Differential flux (1 / (cm2 MeV s sr))
        """
        pix_coord = self.world2pix(lon, lat, energy)
        values = self._interpolate(pix_coord)
        #values.reshape((len(energy), len(lat), len(lon)))
        values.reshape((energy.size, lat.size, lon.size))

        return Quantity(values, '1 / (cm2 MeV s sr)')

    def spectral_index(self, lon, lat, energy, dz=1e-3):
        """Power law spectral index.
        
        A forward finite difference method with step `dz` is used along
        the `z = log10(energy)` axis.

        Parameters
        ----------
        lon : `~astropy.units.Quantity` or `~astropy.units.Angle`
            Longitude
        lat : `~astropy.units.Quantity` or `~astropy.units.Angle`
            Latitude
        energy : `~astropy.units.Quantity`
            Energy

        Returns
        -------
        spectral_index : `~astropy.units.Quantity`
            Spectral index
        """
        # Compute flux at `z = log(energy)`
        pix_coord = self.world2pix(lon, lat, energy)
        flux1 = self._interpolate(pix_coord)

        # Compute flux at `z + dz`
        pix_coord[:, 0] += dz
        # pixel_coordinates += np.zeros(pixel_coordinates.shape)
        flux2 = self._interpolate(pix_coord)

        # Power-law spectral index through these two flux points
        # d_log_flux = np.log(flux2 / flux1)
        # spectral_index = d_log_flux / dz
        energy1 = energy
        energy2 = (1. + dz) * energy
        spectral_index = powerlaw.g_from_points(energy1, energy2, flux1, flux2)

        return spectral_index

    def integral_flux_image(self, energy_band, energy_bins=10):
        """Integral flux image for a given energy band.

        A local power-law approximation in small energy bins is
        used to compute the integral.

        Parameters
        ----------
        energy_band : `astropy.units.Quantity`
            Tuple ``(energy_min, energy_max)``
        energy_bins : int or `astropy.units.Quantity`
            Energy bin definition.
        """
        if isinstance(energy_bins, int):
            energy_bins = energy_bounds_equal_log_spacing(energy_band, energy_bins)

        energy1 = energy_bins[:-1]
        energy2 = energy_bins[1:]
        #lon, lat = self.coordinates()
        #flux = self.flux(lon, lat, energy_bins)
        x, y = np.indices(self.data.shape[1:])
        z = self.energy_axis.world2pix(energy_bins)
        import IPython; IPython.embed(); 1 / 0
        pix_coords = np.dstack((z.flat, y.flat, x.flat))
        flux = self._interpolate(pix_coords)
        flux1 = flux[:-1, :, :]
        flux2 = flux[1:, :, :]

        integral_flux = powerlaw.I_from_points(energy1, energy2, flux1, flux2)
        integral_flux = np.sum(integral_flux, axis=0)

        return integral_flux

    def world2pix(self, lon, lat, energy):
        """Convert world to pixel coordinates.
        
        Parameters
        ----------
        TODO
        
        Returns
        -------
        TODO
        """
        lon = lon.to('deg').value
        lat = lat.to('deg').value

        # We're not interested in the energy axis, so we give a dummy value of 1
        x, y = self.wcs.wcs_world2pix(lon, lat, 1, 0)[:-1]

        #energy = energy.to(self.energy.unit).value
        z = self.energy_axis.world2pix(energy)

        x = np.array(x).flat
        y = np.array(y).flat
        z = np.array(z).flat
        
        return np.dstack([z, y, x]) 

    def pix2idx(self, x, y, z):
        """TODO: is this the right way to do it?
        """
        x = np.round(x).astype(int)
        y = np.round(y).astype(int)
        z = np.round(z).astype(int)
        #z = np.searchsorted(self.energy.value, energy)

        return np.dstack(y.flat, y.flat, x.flat) 

# ******** The following methods are copied from the `spectral-cube` package *************
    @property
    def spatial_coordinate_map(self):
        return self.world[0, :, :][1:]

    def __repr__(self):
        # Copied from `spectral-cube` package
        s = "GammaSpectralCube with shape={0}".format(self.data.shape)
        if self.data.unit is u.dimensionless_unscaled:
            s += ":\n"
        else:
            s += " and unit={0}:\n".format(self.data.unit)
        s += " n_x: {0:5d}  type_x: {1:15s}  unit_x: {2}\n".format(self.data.shape[2], self.wcs.wcs.ctype[0], self.wcs.wcs.cunit[0])
        s += " n_y: {0:5d}  type_y: {1:15s}  unit_y: {2}\n".format(self.data.shape[1], self.wcs.wcs.ctype[1], self.wcs.wcs.cunit[1])
        s += " n_s: {0:5d}  type_s: {1:15s}  unit_s: {2}".format(self.data.shape[0], self.wcs.wcs.ctype[2], self.wcs.wcs.cunit[2])
        return s



'''
Old methods ... can probably all be deleted


    def lookup(self, lon, lat, energy):
        """TODO: Unused ... document or remove.
        """
        x, y = self._get_xy(lon, lat)
        z1, z2, energy1, energy2 = self.e_axis(energy)
        f1, f2 = self.data[z1, y, x], self.data[z2, y, x]
        return [energy1, energy2, f1, f2, energy]

    def _set_position(self, lon, lat):
        """Pre-compute log-flux vector for a given position."""
        x, y = self._get_xy(lon, lat)
        self.log_f = np.log10(self.data[:, y, x])

    def _get_xy(self, lon, lat):
        """Find pixel coordinates for a given position."""
        # We're not interested in the energy axis, so we give a dummy value of 1
        x, y = self.wcs.wcs_world2pix(lon, lat, 1, 0)[:-1]
        # Find the nearest integer pixel
        x = np.round(x).astype(int)
        y = np.round(y).astype(int)
        return x, y
'''
