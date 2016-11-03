# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Gamma-ray spectral cube: longitude, latitude and spectral axis.

TODO: split `SkyCube` into a base class ``SkyCube`` and a few sub-classes:

* ``SkyCube`` to represent functions evaluated at grid points (diffuse model format ... what is there now).
* ``ExposureCube`` should also be supported (same semantics, but different units / methods as ``SkyCube`` (``gtexpcube`` format)
* ``SkyCubeHistogram`` to represent model or actual counts in energy bands (``gtbin`` format)
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict

import numpy as np
from numpy.testing import assert_allclose
from astropy.io import fits
import astropy.units as u
from astropy.units import Quantity
from astropy.table import Table
from astropy.wcs import WCS
from astropy.utils import lazyproperty

from ..utils.scripts import make_path
from ..utils.testing import assert_wcs_allclose
from ..utils.energy import EnergyBounds
from ..utils.fits import table_to_fits_table
from ..image import SkyImage
from ..spectrum import LogEnergyAxis
from ..spectrum.powerlaw import power_law_I_from_points

__all__ = ['SkyCube']


class SkyCube(object):
    """Sky cube with dimensions lon, lat and energy.

    Can be used e.g. for Fermi or GALPROP diffuse model cubes.

    Note that there is a very nice ``SkyCube`` implementation here:
    http://spectral-cube.readthedocs.io/en/latest/index.html

    Here is some discussion if / how it could be used:
    https://github.com/radio-astro-tools/spectral-cube/issues/110

    For now we re-implement what we need here.

    The order of the sky cube axes can be very confusing ... this should help:

    * The ``data`` array axis order is ``(energy, lat, lon)``.
    * The ``wcs`` object is a two dimensional celestial WCS with axis order ``(lon, lat)``.

    Parameters
    ----------
    name : str
        Name of the sky cube.
    data : `~astropy.units.Quantity`
        Data array (3-dim)
    wcs : `~astropy.wcs.WCS`
        Word coordinate system transformation
    energy : `~astropy.units.Quantity`
        Energy array

    Attributes
    ----------
    data : `~astropy.units.Quantity`
        Data array (3-dim)
    wcs : `~astropy.wcs.WCS`
        Word coordinate system transformation
    energy : `~astropy.units.Quantity`
        Energy array
    energy_axis : `~gammapy.spectrum.LogEnergyAxis`
        Energy axis
    meta : `~collections.OrderedDict`
        Dictionary to store meta data.


    Notes
    -----
    Diffuse model files in this format are distributed with the Fermi Science tools
    software and can also be downloaded at
    http://fermi.gsfc.nasa.gov/ssc/data/access/lat/BackgroundModels.html

    E.g. the 2-year diffuse model that was used in the 2FGL catalog production is at
    http://fermi.gsfc.nasa.gov/ssc/data/analysis/software/aux/gal_2yearp7v6_v0.fits
    """

    def __init__(self, name=None, data=None, wcs=None, energy=None, meta=None):
        # TODO: check validity of inputs
        self.name = name
        self.data = data
        self.wcs = wcs
        self.meta = meta

        # TODO: decide whether we want to use an EnergyAxis object or just use the array directly.
        self.energy = energy
        if energy:
            self.energy_axis = LogEnergyAxis(energy)

        self._interpolate_cache = None

    @property
    def _interpolate(self):
        """Interpolated data (`~scipy.interpolate.RegularGridInterpolator`)"""
        if self._interpolate_cache is None:
            # Initialise the interpolator
            # This doesn't do any computations ... I'm not sure if it allocates extra arrays.
            from scipy.interpolate import RegularGridInterpolator
            points = list(map(np.arange, self.data.shape))
            self._interpolate_cache = RegularGridInterpolator(points, self.data.value,
                                                              fill_value=None, bounds_error=False)
        return self._interpolate_cache

    @classmethod
    def read_hdu(cls, hdu_list):
        """Read sky cube from HDU.

        Parameters
        ----------
        object_hdu : `~astropy.io.fits.ImageHDU`
            Image HDU object to be read
        energy_table_hdu : `~astropy.io.fits.TableHDU`
            Table HDU object giving energies of each slice
            of the Image HDU object_hdu

        Returns
        -------
        sky_cube : `SkyCube`
            Sky cube
        """
        object_hdu = hdu_list[0]
        energy_table_hdu = hdu_list[1]
        data = object_hdu.data
        data = Quantity(data, '1 / (cm2 MeV s sr)')
        # Note: the energy axis of the FITS cube is unusable.
        # We only use proj for LON, LAT and call ENERGY from the table extension
        header = object_hdu.header
        wcs = WCS(header)
        energy = energy_table_hdu.data['Energy']
        energy = Quantity(energy, 'MeV')
        meta = OrderedDict(header)
        return cls(data=data, wcs=wcs, energy=energy, meta=meta)

    @classmethod
    def read(cls, filename, format='fermi'):
        """Read sky cube from FITS file.

        Parameters
        ----------
        filename : str
            File name
        format : {'fermi', 'fermi-counts'}
            Fits file format.

        Returns
        -------
        sky_cube : `SkyCube`
            Sky cube
        """
        filename = str(make_path(filename))
        data = fits.getdata(filename)
        # Note: the energy axis of the FITS cube is unusable.
        # We only use proj for LON, LAT and do ENERGY ourselves
        header = fits.getheader(filename)
        wcs = WCS(header).celestial
        meta = OrderedDict(header)
        if format == 'fermi':
            energy = Table.read(filename, 'ENERGIES')['Energy']
            energy = Quantity(energy, 'MeV')
            data = Quantity(data, '1 / (cm2 MeV s sr)')
        elif format == 'fermi-counts':
            energy = EnergyBounds.from_ebounds(fits.open(filename)['EBOUNDS'], unit='keV')
            data = Quantity(data, 'count')
        else:
            raise ValueError('Not a valid cube fits format')

        return cls(data=data, wcs=wcs, energy=energy, meta=meta)

    def fill_events(self, events, weights=None):
        """
        Fill events (modifies ``data`` attribute).

        Parameters
        ----------
        events : `~gammapy.data.EventList`
            Event list
        weights : str, optional
            Column to use as weights (none by default)
        """
        if weights is not None:
            weights = events[weights]

        xx, yy, zz = self.wcs_skycoord_to_pixel(events.radec, events.energy)
       
        bins = self._bins_energy, self.spatial._bins_pix[0], self.spatial._bins_pix[1]
        data = np.histogramdd([zz, yy, xx], bins, weights=weights)[0]

        self.data = self.data + data

    @property
    def _bins_energy(self):
        return np.arange(self.data.shape[0] + 1)

    @classmethod
    def empty(cls, emin=0.5, emax=100, enbins=10, eunit='TeV', **kwargs):
        """
        Create empty sky cube with log equal energy binning from the scratch.

        Parameters
        ----------
        emin : float
            Minimum energy.
        emax : float
            Maximum energy.
        enbins : int
            Number of energy bins.
        eunit : str
            Energy unit.
        kwargs : dict
            Keyword arguments passed to `~gammapy.image.SkyImage.empty` to create
            the spatial part of the cube.
        """
        image = SkyImage.empty(**kwargs)
        energy = EnergyBounds.equal_log_spacing(emin, emax, enbins, eunit)
        data = image.data * np.ones(len(energy)).reshape((-1, 1, 1))
        return cls(data=data, wcs=image.wcs, energy=energy)

    @classmethod
    def empty_like(cls, refcube, fill=0):
        """
        Create an empty sky cube with the same WCS, energy specification and meta
        as given sky cube.

        Parameters
        ----------
        refcube : `~gammapy.cube.SkyCube`
            Reference sky cube.
        fill : float, optional
            Fill image with constant value. Default is 0.
        """
        wcs = refcube.wcs.copy()
        data = fill * np.ones_like(refcube.data)
        energies = refcube.energies.copy()
        return cls(data=data, wcs=wcs, energy=energies, meta=refcube.meta)

    def wcs_skycoord_to_pixel(self, position, energy):
        """Convert world to pixel coordinates.

        Parameters
        ----------
        position : `~astropy.coordinates.SkyCoord`
            Position on the sky.
        energy : `~astropy.units.Quantity`
            Energy
        
        Returns
        -------
        (x, y, z) : tuple
            Tuple of x, y, z coordinates.
        """
        if not position.shape == energy.shape:
            raise ValueError('Position and energy array must have the same shape.')

        x, y = self.spatial.wcs_skycoord_to_pixel(position)
        z = self.energy_axis.world2pix(energy)
        
        #TODO: check order, so that it corresponds to data axis order
        return (x, y, z)

    def wcs_pixel_to_skycoord(self, x, y, z):
        """Convert pixel to world coordinates.

        Parameters
        ----------
        x, y, z

        Returns
        -------
        lon, lat, energy
        """
        position = self.spatial.wcs_pixel_to_skycoord(x, y)
        energy = self.energy_axis.pix2world(z)
        energy = Quantity(energy, self.energy.unit)
        return (position, energy)

    def to_sherpa_data3d(self):
        """
        Convert sky cube to sherpa `Data3D` object.
        """
        from .sherpa_ import Data3D

        # Energy axes
        energies = self.energy.to("TeV").value
        ebounds = EnergyBounds(Quantity(energies, 'TeV'))
        elo = ebounds.lower_bounds.value
        ehi = ebounds.upper_bounds.value

        coordinates = self.spatial.coordinates()
        ra = coordinates.data.lon
        dec = coordinates.data.lat

        n_ebins = len(elo)
        ra_cube = np.tile(ra, (n_ebins, 1, 1))
        dec_cube = np.tile(dec, (n_ebins, 1, 1))
        elo_cube = elo.reshape(n_ebins, 1, 1) * np.ones_like(ra)
        ehi_cube = ehi.reshape(n_ebins, 1, 1) * np.ones_like(ra)

        return Data3D('', elo_cube.ravel(), ehi_cube.ravel(), ra_cube.ravel(),
                      dec_cube.ravel(), self.data.value.ravel(),
                      self.data.value.shape)

    def sky_image(self, idx_energy, copy=True):
        """Slice a 2-dim `~gammapy.image.SkyImage` from the cube.

        Parameters
        ----------
        idx_energy : int
            Energy slice index
        copy : bool (default True)
            Whether to make deep copy of returned object

        Returns
        -------
        image : `~gammapy.image.SkyImage`
            2-dim sky image
        """
        # TODO: should we pass something in SkyImage (we speak about meta)?
        data = self.data[idx_energy]
        image = SkyImage(name=self.name, data=data, wcs=self.wcs)
        return image.copy() if copy else image

    @lazyproperty        
    def spatial(self):
        """
        Spatial part of the cube obtained by summing over all energy bins.

        Examples
        --------
        Can be used to acces the spatial information of the cube:

            >>> from gammapy.cube import SkyCube
            >>> cube = SkyCube.empty()
            >>> coords = cube.spatial.coordinates()
            >>> solid_angle = cube.spatial.solid_angle()

        """
        #TODO: what about meta info?
        data = np.nansum(self.data, axis=0)
        wcs = self.wcs.celestial.copy()
        return SkyImage(name=self.name, data=data, wcs=wcs)

    @lazyproperty        
    def spectral(self):
        """
        Spectral part of the cube obtained by summing over all spatial bins.

        """
        from ..spectrum import CountsSpectrum
        #TODO: what about meta info?
        data = np.nansum(np.nansum(self.data, axis=1), axis=1)
        return CountsSpectrum(data=data, energy=self.energy)

    def lookup(self, position, energy, interpolation=False):
        """Differential flux.

        Parameters
        ----------
        position : `~astropy.coordinates.SkyCoord`
            Position on the sky.
        energy : `~astropy.units.Quantity`
            Energy

        Returns
        -------
        flux : `~astropy.units.Quantity`
            Differential flux (1 / (cm2 MeV s sr))
        """
        # TODO: add interpolation option using NDDataArray
        if not position.shape == energy.shape:
            raise ValueError('Position and energy array must have the same shape.')

        z, y, x = self.wcs_skycoord_to_pixel(position, energy)
       
        if interpolation:
            shape = z.shape
            pix_coords = np.column_stack([x.flat, y.flat, z.flat])
            vals = self._interpolate(pix_coords)
            return vals.reshape(shape)
        else:
            return self.data[np.rint(z).astype('int'), np.rint(y).astype('int'),
                             np.rint(x).astype('int')]

    def show(self, viewer='mpl', ds9options=None, **kwargs):
        """
        Show sky cube in image viewer.

        Parameters
        ----------
        viewer : {'mpl', 'ds9'}
            Which image viewer to use. Option 'ds9' requires ds9 to be installed.
        ds9options : list, optional
            List of options passed to ds9. E.g. ['-cmap', 'heat', '-scale', 'log'].
            Any valid ds9 command line option can be passed.
            See http://ds9.si.edu/doc/ref/command.html for details.
        **kwargs : dict
            Keyword arguments passed to `~matplotlib.pyplot.imshow`.
        """
        import matplotlib.pyplot as plt
        from ipywidgets import interact

        if viewer == 'mpl':
            max_ = self.data.shape[0] - 1
            
            def show_image(idx):
                image = self.sky_image(idx)
                image.data = image.data.value
                image.show(**kwargs)

            return interact(show_image, idx=(0, max_, 1))
        elif viewer == 'ds9':
            raise NotImplementedError

    def integral_flux_image(self, energy_band, energy_bins=10):
        """Integral flux image for a given energy band.

        A local power-law approximation in small energy bins is
        used to compute the integral.

        Parameters
        ----------
        energy_band : `~astropy.units.Quantity`
            Tuple ``(energy_min, energy_max)``
        energy_bins : int or `~astropy.units.Quantity`
            Energy bin definition.

        Returns
        -------
        image : `~gammapy.image.SkyImage`
            Integral flux image (1 / (cm^2 s sr))
        """
        if isinstance(energy_bins, int):
            energy_bins = EnergyBounds.equal_log_spacing(
                energy_band[0], energy_band[1], energy_bins)
        else:
            energy_bins = EnergyBounds(energy_band)

        energy_bins = energy_bins.to('MeV')
        energy1 = energy_bins.lower_bounds
        energy2 = energy_bins.upper_bounds

        # Compute differential flux at energy bin edges of all pixels
        xx = np.arange(self.data.shape[2])
        yy = np.arange(self.data.shape[1])
        zz = self.energy_axis.world2pix(energy_bins)

        xx, yy, zz = np.meshgrid(zz, yy, xx, indexing='ij')
        shape = xx.shape

        pix_coords = np.column_stack([xx.flat, yy.flat, zz.flat])
        flux = self._interpolate(pix_coords)
        flux = flux.reshape(shape)

        # Compute integral flux using power-law approximation in each bin
        flux1 = flux[:-1, :, :]
        flux2 = flux[1:, :, :]
        energy1 = energy1[:, np.newaxis, np.newaxis].value
        energy2 = energy2[:, np.newaxis, np.newaxis].value
        integral_flux = power_law_I_from_points(energy1, energy2, flux1, flux2)

        integral_flux = integral_flux.sum(axis=0)

        header = self.wcs.to_header()

        image = SkyImage(name='flux',
                         data=integral_flux,
                         wcs=self.wcs,
                         unit='cm^-2 s^-1 sr^-1',
                         meta=header)
        return image

    def reproject(self, reference, mode='interp', *args, **kwargs):
        """Spatially reprojects a `SkyCube` onto a reference.

        Parameters
        ----------
        reference : `~astropy.io.fits.Header`, `SkyImage` or `SkyCube`
            Reference wcs specification to reproject the data on.
        mode : {'interp', 'exact'}
            Interpolation mode.
        *args : list
            Arguments passed to `~reproject.reproject_interp` or
            `~reproject.reproject_exact`.
        **kwargs : dict
            Keyword arguments passed to `~reproject.reproject_interp` or
            `~reproject.reproject_exact`.

        Returns
        -------
        reprojected_cube : `SkyCube`
            Cube spatially reprojected to the reference.
        """
        if isinstance(reference, SkyCube):
            reference = reference.spatial

        out = []
        for idx in range(len(self.data)):
            image = self.sky_image(idx)
            image_out = image.reproject(reference, mode=mode, *args, **kwargs)
            out.append(image_out.data)

        data = Quantity(np.stack(out, axis=0), self.data.unit)
        wcs = image_out.wcs.copy()
        return self.__class__(name=self.name, data=data, wcs=wcs, meta=self.meta,
                              energy=self.energy)

 
    def to_fits(self):
        """Writes SkyCube to FITS hdu_list.

        Returns
        -------
        hdu_list : `~astropy.io.fits.HDUList`
            * hdu_list[0] : `~astropy.io.fits.PrimaryHDU`
                Image array of data
            * hdu_list[1] : `~astropy.io.fits.BinTableHDU`
                Table of energies
        """
        image = fits.PrimaryHDU(self.data.value, self.wcs.to_header())
        image.header['SPECUNIT'] = '{0.unit:FITS}'.format(self.data)

        # for BinTableHDU's the data must be added via a Table object
        energy_table = Table()
        energy_table['Energy'] = self.energy
        energy_table.meta['name'] = 'ENERGY'

        energies = table_to_fits_table(energy_table)

        hdu_list = fits.HDUList([image, energies])

        return hdu_list

    def to_images(self):
        """Convert to `~gammapy.cube.SkyCubeImages`.
        """
        from .images import SkyCubeImages
        images = [self.sky_image(idx) for idx in range(len(self.data))]
        return SkyCubeImages(self.name, images, self.wcs, self.energy)

    def write(self, filename, **kwargs):
        """Write to FITS file.

        Parameters
        ----------
        filename : str
            Filename
        """
        filename = str(make_path(filename))
        self.to_fits().writeto(filename, **kwargs)

    def __repr__(self):
        # Copied from `spectral-cube` package
        ss = "Sky cube {} with shape={}".format(self.name, self.data.shape)
        if self.data.unit is u.dimensionless_unscaled:
            ss += ":\n"
        else:
            ss += " and unit={}:\n".format(self.data.unit)

        ss += " n_lon:    {:5d}  type_lon:    {:15s}  unit_lon:    {}\n".format(
            self.data.shape[2], self.wcs.wcs.ctype[0], self.wcs.wcs.cunit[0])
        ss += " n_lat:    {:5d}  type_lat:    {:15s}  unit_lat:    {}\n".format(
            self.data.shape[1], self.wcs.wcs.ctype[1], self.wcs.wcs.cunit[1])
        ss += " n_energy: {:5d}  unit_energy: {}\n".format(
            len(self.energy), self.energy.unit)

        return ss

    def info(self):
        """
        Print summary info about the cube.
        """
        print(repr(self))

    @staticmethod
    def assert_allclose(cube1, cube2):
        assert cube1.name == cube2.name
        assert_allclose(cube1.data, cube2.data)
        assert_wcs_allclose(cube1.wcs, cube2.wcs)
