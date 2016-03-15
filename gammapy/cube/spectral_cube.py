# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Gamma-ray spectral cube: longitude, latitude and spectral axis.

TODO: split `SpectralCube` into a base class ``SpectralCube`` and a few sub-classes:

* ``SpectralCube`` to represent functions evaluated at grid points (diffuse model format ... what is there now).
* ``ExposureCube`` should also be supported (same semantics, but different units / methods as ``SpectralCube`` (``gtexpcube`` format)
* ``SpectralCubeHistogram`` to represent model or actual counts in energy bands (``gtbin`` format)
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from astropy.io import fits
import astropy.units as u
from astropy.units import Quantity
from astropy.table import Table
from astropy.wcs import WCS

from ..spectrum import LogEnergyAxis, powerlaw
from ..utils.energy import EnergyBounds
from ..image import cube_to_image, solid_angle, SkyMap
from ..image.utils import _bin_events_in_cube 
from ..utils.fits import table_to_fits_table
from ..utils.wcs import get_wcs_ctype

__all__ = ['SpectralCube']


class SpectralCube(object):
    """Spectral cube for gamma-ray astronomy.

    Can be used e.g. for Fermi or GALPROP diffuse model cubes.

    Note that there is a very nice ``SpectralCube`` implementation here:
    http://spectral-cube.readthedocs.org/en/latest/index.html

    Here is some discussion if / how it could be used:
    https://github.com/radio-astro-tools/spectral-cube/issues/110

    For now we re-implement what we need here.

    The order of the spectral cube axes can be very confusing ... this should help:

    * The ``data`` array axis order is ``(energy, lat, lon)``.
    * The ``wcs`` object axis order is ``(lon, lat, energy)``.
    * Methods use the ``wcs`` order of ``(lon, lat, energy)``,
      but internally when accessing the data often the reverse order is used.
      We use ``(xx, yy, zz)`` as pixel coordinates for ``(lon, lat, energy)``,
      as that matches the common definition of ``x`` and ``y`` in image viewers.

    Parameters
    ----------
    name : str
        Name of the spectral cube.
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

    Notes
    -----
    Diffuse model files in this format are distributed with the Fermi Science tools
    software and can also be downloaded at
    http://fermi.gsfc.nasa.gov/ssc/data/access/lat/BackgroundModels.html

    E.g. the 2-year diffuse model that was used in the 2FGL catalog production is at
    http://fermi.gsfc.nasa.gov/ssc/data/analysis/software/aux/gal_2yearp7v6_v0.fits
    """

    def __init__(self, name=None, data=None, wcs=None, energy=None):
        # TODO: check validity of inputs
        self.name = name
        self.data = data
        self.wcs = wcs
        
        # TODO: decide whether we want to use an EnergyAxis object or just use the array directly.
        self.energy = energy
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
        """Read spectral cube from HDU.

        Parameters
        ----------
        object_hdu : `~astropy.io.fits.ImageHDU`
            Image HDU object to be read
        energy_table_hdu : `~astropy.io.fits.TableHDU`
            Table HDU object giving energies of each slice
            of the Image HDU object_hdu

        Returns
        -------
        spectral_cube : `SpectralCube`
            Spectral cube
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
        return cls(data, wcs, energy)

    @classmethod
    def read(cls, filename, format='fermi'):
        """Read spectral cube from FITS file.

        Parameters
        ----------
        filename : str
            File name
        format : {'fermi', 'fermi-counts'}
            Fits file format.

        Returns
        -------
        spectral_cube : `SpectralCube`
            Spectral cube
        """
        data = fits.getdata(filename)
        # Note: the energy axis of the FITS cube is unusable.
        # We only use proj for LON, LAT and do ENERGY ourselves
        header = fits.getheader(filename)
        wcs = WCS(header).celestial
        if format == 'fermi':
            energy = Table.read(filename, 'ENERGIES')['Energy']
            energy = Quantity(energy, 'MeV')
            data = Quantity(data, '1 / (cm2 MeV s sr)')
        elif format == 'fermi-counts':
            energy = EnergyBounds.from_ebounds(fits.open(filename)['EBOUNDS'], unit='keV')
            data = Quantity(data, 'counts')
        else:
            raise ValueError('Not a valid cube fits format')
        return cls(data=data, wcs=wcs, energy=energy)

    def fill(self, events, origin=0):
        """
        Fill spectral cube with events.

        Parameters
        ----------
        events : `~astropy.table.Table`
            Event list table
        origin : {0, 1}
            Pixel coordinate origin.

        """
        self.data = _bin_events_in_cube(events, self.wcs, self.data.shape, self.energy)
        
    @classmethod
    def empty(cls, emin=0.5, emax=100, enbins=10, eunit='TeV', **kwargs):
        """
        Create empty spectral cube with log equal energy binning from the scratch.
        
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
            Keyword arguments passed to `~gammapy.image.SkyMap.empty` to create
            the spatial part of the cube.
        """
        refmap = SkyMap.empty(**kwargs)
        energies = EnergyBounds.equal_log_spacing(emin, emax, enbins, eunit)
        data = refmap.data * np.ones(len(energies)).reshape((-1, 1, 1))
        return cls(data=data, wcs=refmap.wcs, energy=energies)

    @classmethod
    def empty_like(cls, refcube, fill=0):
        """
        Create an empty spectral cube with the same WCS and energy specification
        as given spectral cube. 
        
        Parameters
        ----------
        refcube : `~gammapy.cube.SpectralCube`
            Reference spectral cube.
        fill : float, optional
            Fill sky map with constant value. Default is 0.
        """
        wcs = refcube.wcs.copy()
        data = fill * np.ones_like(refcube.data)
        energies = refcube.energies.copy()
        return cls(data=data, wcs=wcs, energy=energies)        

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
        x, y, _ = self.wcs.wcs_world2pix(lon, lat, 0, origin)

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

    @property
    def spatial_coordinate_images(self):
        """Spatial coordinate images (2x `~astropy.units.Quantity`)

        Returns two separate objects for the arrays of longitude
        and latitude pixel coordinates.
        """
        n_lon = self.data.shape[2]
        n_lat = self.data.shape[1]
        i_lat, i_lon = np.indices((n_lat, n_lon))
        lon, lat, _ = self.pix2world(i_lon, i_lat, 0)

        return lon, lat

    def to_sherpa_data3d(self):
        """
        Convert spectral cube to sherpa `Data3D` object.
        """
        from .sherpa_ import Data3D

        # Energy axes
        energies = self.energy.to("TeV").value
        ebounds = EnergyBounds(Quantity(energies, 'TeV'))
        elo = ebounds.lower_bounds.value
        ehi = ebounds.upper_bounds.value  

        ra, dec = self.spatial_coordinate_images
       
        n_ebins = len(elo)
        ra_cube = np.tile(ra, (n_ebins, 1, 1))
        dec_cube = np.tile(dec, (n_ebins, 1, 1))
        elo_cube = elo.reshape(n_ebins, 1, 1) * np.ones_like(ra)
        ehi_cube = ehi.reshape(n_ebins, 1, 1) * np.ones_like(ra)
 
        return Data3D('', elo_cube.ravel(), ehi_cube.ravel(), ra_cube.ravel(),    
                      dec_cube.ravel(), counts.data.value.ravel(),
                      counts.data.value.shape)


    @property
    def solid_angle_image(self):
        """Solid angle image in steradian (`~astropy.units.Quantity`)"""
        cube_hdu = fits.ImageHDU(self.data, self.wcs.to_header())
        image_hdu = cube_to_image(cube_hdu)
        image_hdu.header['WCSAXES'] = 2

        return solid_angle(image_hdu).to('sr')


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
        image : `~astropy.io.fits.ImageHDU`
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
        integral_flux = powerlaw.I_from_points(energy1, energy2, flux1, flux2)

        integral_flux = integral_flux.sum(axis=0)

        # TODO: get rid of the `str` calls once this `WCS.sub` issue is fixed:
        # https://github.com/astropy/astropy/issues/3356
        axes = [str('longitude'), str('latitude')]
        header = self.wcs.sub(axes).to_header()

        hdu = fits.ImageHDU(data=integral_flux,
                            header=header, name='integral_flux')

        return hdu

    def reproject_to(self, reference_cube, projection_type='bicubic'):
        """
        Spatially reprojects a `SpectralCube` onto a reference cube.

        Parameters
        ----------
        reference_cube : `SpectralCube`
            Reference cube with the desired spatial projection.
        projection_type : {'nearest-neighbor', 'bilinear',
            'biquadratic', 'bicubic', 'flux-conserving'}
            Specify method of reprojection. Default: 'bilinear'.

        Returns
        -------
        reprojected_cube : `SpectralCube`
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

        wcs_out = WCS(header_out)

        return SpectralCube(new_cube, wcs_out, energy)

    def to_fits(self):
        """Writes SpectralCube to FITS hdu_list.

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

    def writeto(self, filename, **kwargs):
        """Writes SpectralCube to FITS file.

        Parameters
        ----------
        filename : string
            Name of output file (.fits)
        """
        self.to_fits().writeto(filename, **kwargs)

    def __repr__(self):
        # Copied from `spectral-cube` package
        s = "Spectral cube {} with shape={0}".format(self.name, self.data.shape)
        if self.data.unit is u.dimensionless_unscaled:
            s += ":\n"
        else:
            s += " and unit={0}:\n".format(self.data.unit)
        s += " n_x: {0:5d}  type_x: {1:15s}  unit_x: {2}\n".format(self.data.shape[2], self.wcs.wcs.ctype[0],
                                                                   self.wcs.wcs.cunit[0])
        s += " n_y: {0:5d}  type_y: {1:15s}  unit_y: {2}\n".format(self.data.shape[1], self.wcs.wcs.ctype[1],
                                                                   self.wcs.wcs.cunit[1])
        s += " n_s: {0:5d}  type_s: {1:15s}  unit_s: {2}".format(self.data.shape[0], self.wcs.wcs.ctype[2],
                                                                 self.wcs.wcs.cunit[2])
        return s


    def info(self):
        """
        Print summary info about the cube.
        """
        print(repr(self))