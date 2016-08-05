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
from astropy.io import fits
import astropy.units as u
from astropy.units import Quantity
from astropy.table import Table
from astropy.wcs import WCS
from ..utils.energy import EnergyBounds
from ..utils.fits import table_to_fits_table
from ..image import SkyImage
from ..image.utils import _bin_events_in_cube
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
    meta : '~collections.OrderedDict'
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

    def fill(self, events, origin=0):
        """
        Fill sky cube with events.

        Parameters
        ----------
        events : `~astropy.table.Table`
            Event list table
        origin : {0, 1}
            Pixel coordinate origin.
        """
        self.data = _bin_events_in_cube(events, self.wcs, self.data.shape, self.energy, origin=origin)

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
        refmap = SkyImage.empty(**kwargs)
        energy = EnergyBounds.equal_log_spacing(emin, emax, enbins, eunit)
        data = refmap.data * np.ones(len(energy)).reshape((-1, 1, 1))
        return cls(data=data, wcs=refmap.wcs, energy=energy)

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

    def world2pix(self, lon, lat, energy, combine=False):
        """Convert world to pixel coordinates.

        Parameters
        ----------
        lon, lat, energy

        Returns
        -------
        x, y, z or array with (x, y, z) as columns
        """
        lon = lon.to('deg').value
        lat = lat.to('deg').value
        origin = 0  # convention for gammapy
        x, y = self.wcs.wcs_world2pix(lon, lat, origin)

        z = self.energy_axis.world2pix(energy)

        shape = (x * y * z).shape
        x = x * np.ones(shape)
        y = y * np.ones(shape)
        z = z * np.ones(shape)

        if combine:
            x = np.array(x).flat
            y = np.array(y).flat
            z = np.array(z).flat
            return np.column_stack([z, y, x])
        else:
            return x, y, z

    def pix2world(self, x, y, z):
        """Convert world to pixel coordinates.

        Parameters
        ----------
        x, y, z

        Returns
        -------
        lon, lat, energy
        """
        origin = 0  # convention for gammapy
        lon, lat = self.wcs.wcs_pix2world(x, y, origin)
        energy = self.energy_axis.pix2world(z)

        lon = Quantity(lon, 'deg')
        lat = Quantity(lat, 'deg')
        energy = Quantity(energy, self.energy.unit)

        return lon, lat, energy

    def coordinates(self, mode='center'):
        """Spatial coordinate images.

        Wrapper of `gammapy.image.SkyImage.coordinates`

        Parameters
        ----------
        mode : {'center', 'edges'}
            Return coordinate values at the pixels edges or pixel centers.

        Returns
        -------
        coordinates : `~astropy.coordinates.SkyCoord`
            Position on the sky.
        """
        image = self.sky_image(0)
        coordinates = image.coordinates(mode)
        return coordinates

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

        coordinates = self.coordinates()
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

    @property
    def solid_angle(self):
        """Solid angle image in steradian (`~astropy.units.Quantity`)"""
        image = self.sky_image(idx_energy=0)
        return image.solid_angle()

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
        data = Quantity(self.data[idx_energy], self.data.unit)
        image = SkyImage(name=self.name, data=data, wcs=self.wcs)
        return image.copy() if copy else image

    def flux(self, lon, lat, energy):
        """Differential flux.

        Parameters
        ----------
        lon : `~astropy.coordinates.Angle`
            Longitude
        lat : `~astropy.coordinates.Angle`
            Latitude
        energy : `~astropy.units.Quantity`
            Energy

        Returns
        -------
        flux : `~astropy.units.Quantity`
            Differential flux (1 / (cm2 MeV s sr))
        """
        # Determine output shape by creating some array via broadcasting
        shape = (lon * lat * energy).shape

        pix_coord = self.world2pix(lon, lat, energy, combine=True)
        values = self._interpolate(pix_coord)
        values = values.reshape(shape)

        return Quantity(values, '1 / (cm2 MeV s sr)')

    def spectral_index(self, lon, lat, energy, dz=1e-3):
        """Power law spectral index (`numpy.array`).

        A forward finite difference method with step ``dz`` is used along
        the ``z = log10(energy)`` axis.

        Parameters
        ----------
        lon : `~astropy.coordinates.Angle`
            Longitude
        lat : `~astropy.coordinates.Angle`
            Latitude
        energy : `~astropy.units.Quantity`
            Energy
        """
        raise NotImplementedError
        # Compute flux at `z = log(energy)`
        pix_coord = self.world2pix(lon, lat, energy, combine=True)
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

    def reproject_to(self, reference_cube, projection_type='bicubic'):
        """Spatially reprojects a `SkyCube` onto a reference cube.

        Parameters
        ----------
        reference_cube : `SkyCube`
            Reference cube with the desired spatial projection.
        projection_type : {'nearest-neighbor', 'bilinear',
            'biquadratic', 'bicubic', 'flux-conserving'}
            Specify method of reprojection. Default: 'bilinear'.

        Returns
        -------
        reprojected_cube : `SkyCube`
            Cube spatially reprojected to the reference cube.
        """
        from reproject import reproject_interp

        reference = reference_cube.data
        shape_out = reference[0].shape
        try:
            wcs_in = self.wcs.dropaxis(2)
        except:
            wcs_in = self.wcs
        try:
            wcs_out = reference_cube.wcs.dropaxis(2)
        except:
            wcs_out = reference_cube.wcs
        energy = self.energy

        cube = self.data
        new_cube = np.zeros((cube.shape[0], reference.shape[1],
                             reference.shape[2]))
        energy_slices = np.arange(cube.shape[0])
        # TODO: Re-implement to reproject cubes directly without needing
        # to loop over energies here. Errors with reproject when doing this
        # first need to be understood and fixed.
        for i in energy_slices:
            array = cube[i]
            data_in = (array.value, wcs_in)
            new_cube[i] = reproject_interp(data_in, wcs_out, shape_out, order=projection_type)[0]
        new_cube = Quantity(new_cube, array.unit)
        # Create new wcs
        header_in = self.wcs.to_header()
        header_out = reference_cube.wcs.to_header()
        # Keep output energy info the same as input, but changes spatial information
        # So need to restore energy parameters to input values here
        try:
            header_out['CRPIX3'] = header_in['CRPIX3']
            header_out['CDELT3'] = header_in['CDELT3']
            header_out['CTYPE3'] = header_in['CTYPE3']
            header_out['CRVAL3'] = header_in['CRVAL3']
        except:
            pass

        wcs_out = WCS(header_out).celestial

        # TODO: how to fill 'meta' in better way?
        meta = OrderedDict(header_out)
        return SkyCube(data=new_cube, wcs=wcs_out, energy=energy, meta=meta)

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

    def to_image_list(self):
        """Convert sky cube to a `gammapy.image.SkyImageList`.
        """
        from ..image.lists import SkyImageList
        images = [self.sky_image(idx) for idx in range(len(self.data))]
        return SkyImageList(self.name, images, self.wcs, self.energy)

    def writeto(self, filename, **kwargs):
        """Writes SkyCube to FITS file.

        Parameters
        ----------
        filename : str
            Filename
        """
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
        ss += " n_energy: {:5d}  unit_energy: {}".format(
            len(self.energy), self.energy.unit)

        return ss

    def info(self):
        """
        Print summary info about the cube.
        """
        print(repr(self))
