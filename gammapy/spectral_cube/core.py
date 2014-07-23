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
from ..image.utils import coordinates
from ..irf import EnergyDependentTablePSF
from astropy.coordinates import Angle
from scipy.ndimage import convolve
from ..image import cube_to_image


__all__ = ['GammaSpectralCube', 'compute_npred_cube']


#def _correlate_fermi_psf(image, max_offset, resolution=1,
#                         energy='None', energy_band=[10, 500]):
#    from ..datasets import FermiGalacticCenter
#    filename = FermiGalacticCenter.filenames()['psf']
#    pixel_size = Angle(resolution, 'deg')
#    offset_max = Angle(max_offset, 'deg')
#    if energy == 'None':
#        fermi_psf = EnergyDependentTablePSF.read(filename)
#        # PSF energy band calculation doesn't work, so implemented at log center energy instead
#        energy = Quantity(np.sqrt(energy_band[0] * energy_band[1]), 'MeV')
#        psf = fermi_psf.table_psf_at_energy(energy)
#    else:
#        energy = Quantity(energy, 'MeV')
#        fermi_psf = EnergyDependentTablePSF.read(filename)
#        psf = fermi_psf.table_psf_at_energy(energy=energy)
#    psf.normalize()
#    kernel = psf.kernel(pixel_size=pixel_size, offset_max=offset_max)
#    kernel_image = kernel.value / kernel.value.sum()
#    return convolve(image, kernel_image, mode='constant')#


#def _interp_flux(hdu_list, new_energy, method='linear'):
#    hdu = hdu_list[0]
#    cube = GammaSpectralCube.read_hdu(hdu_list)
#    image_hdu = cube_to_image(hdu_list[0], 0)
#    lat, lon = coordinates(image_hdu, world=True, radians=True)
#    lat = Quantity(lat, 'rad')
#    lon = Quantity(lon, 'rad')
#    if method == 'linear':
#        array = cube.flux(lat, lon, new_energy.to('MeV'))
#    elif method == 'log10':
#        # Needs to go into cube.flux as linear relation
#        array = cube.flux(lat, lon, Quantity(np.log10(new_energy), 'MeV'))
#    elif method == 'log':
#        # Needs to go into cube.flux as linear relation
#        array = cube.flux(lat, lon, Quantity(np.log(new_energy), 'MeV'))
#    array = array.reshape(lat.shape)
#    out_hdu = fits.ImageHDU(data = array, header = hdu.header)
#    new_table = hdu_list[1].copy()
#    # True (linear) values are provided in the table
#    new_table.data['Energy'] = new_energy
#    return [out_hdu, new_table]


#def _interp_exposure(hdu_list, new_energy):
#    max_energy = max(hdu_list[1].data['Energy'])
#    new_energy = new_energy.to('MeV')
#    if new_energy.value >= max_energy:
#        max_index = len(hdu_list[1].data['Energy'])
#        hdu = hdu_list[0]
#        a = hdu.data[max_index - 1]
#        out_hdu = fits.ImageHDU(data = a, header = hdu_list[0].header)
#        new_table = hdu_list[1].copy()
#        new_table.data['Energy'] = new_energy
#        return [out_hdu, new_table]
#    else:
#        cube = GammaSpectralCube.read_hdu(hdu_list)
#        image_hdu = cube_to_image(hdu_list[0], 0)
#        lat, lon = coordinates(image_hdu, world=True, radians=True)#
#        lat = Quantity(lat, 'rad')
#        lon = Quantity(lon, 'rad')
#        # This doesn't appear to work - just returns the same each time
#        array = cube.flux(lat, lon, new_energy)#

#        a = array.reshape(lat.shape)
#        out_hdu = fits.ImageHDU(data = a, header = hdu_list[0].header)
#        new_table = hdu_list[1].copy()
#        new_table.data['Energy'] = new_energy
#        return [out_hdu, new_table]


#def _equate_energies(hdu_list1, hdu_list2, energies=None):
#    """Interpolates assuming power law the energy axis of hdu1, and returns cube
#    with energy slices of hdu2.
#    """
#    hdu1 = hdu_list1[0]
#    hdu2 = hdu_list2[0]
#    energies1 = hdu_list1[1].data['Energy']
#    energies2 = hdu_list2[1].data['Energy']
#    if energies == None:
#        indices = np.arange(len(energies2))
#        out_hdu = hdu1.copy()
#        # Only need to change the size of the energy axis to be the same as hdu2
#        out_hdu.data = np.zeros((hdu2.data.shape[0], hdu1.data.shape[1],
#                                 hdu1.data.shape[2]))
#        for index in indices:
#            energy = energies2[index]
#            desired_energy = Quantity(energy, 'MeV')
#            slice_hdus = _interp_flux(hdu_list1, desired_energy)
#            # -1 due to different indexing convention
#            out_hdu.data[index] = slice_hdus[0].data
#        return [out_hdu, hdu_list2[1]]
#    else:
#        indices = len(energies)
#        for index in indices:
#            energy = energies[index]
#            desired_energy = Quantity(energy, 'MeV')
#            slice_hdus = _interp_flux(hdu1, desired_energy)
#            # -1 due to different indexing convention
#            out_hdu.data[index - 1] = slice_hdus[0].data
#            out_energies = hdu_list1[1].copy()
#            out_energies.data['Energy'] = energies
#        return [out_hdu, out_energies]#


#def _reproject_cube(hdu_list1, hdu_list2, smooth=False):
#    """Reprojects hdu1 to the header of hdu2 and returns as hdu.

#    Optionally smooths HDUs to match resolution.
#    """
#    from reproject.interpolation import interpolate_2d
#    out_hdu_list = _equate_energies(hdu_list1, hdu_list2)
#    array = out_hdu_list[0].data
#    wcs_in = WCS(cube_to_image(hdu_list1[0]).header)
#    wcs_out = WCS(cube_to_image(hdu_list2[0]).header)
#    shape_out = cube_to_image(hdu_list2[0]).data.shape
#    energies = out_hdu_list[1].data['Energy']
#    indices = np.arange(len(energies))
#    out_array = np.zeros_like(hdu_list2[0].data)
#    for index in indices:
#        out_array[index] = interpolate_2d(array[index], wcs_in, wcs_out, shape_out, order=3)
#    out_hdu = fits.ImageHDU(data=out_array, header=out_hdu_list[0].header)
#    return [out_hdu, out_hdu_list[1]]#


#def convolve_npred_cube(npred_cube, psf):
#    pass

#def compute_npred_cube_simple(flux_cube, exposure_cube, energy_bin_edges):
#
#    # desired energy binning for the output npred cube
#    energy_bin_edges = 'TODO' # the bin edges
#    energy_bin_centers = np.diff(energy_bin_edges)#

    # desired spatial binning assumed to be the same as for expoure cube
    #lon, lat = exposure_cube.spatial_coordinates

#    solid_angle = exposure_cube.solid_angle # in steradian

#    exposure = exposure_cube.flux()


 #   if method = 'method1':
 #       int_fluxes = []
 #       for ii in range(len(enegy_bin_edges) - 1):
 #           energy_bin = energy_bin_edges[i], energy_bin_edges[i + 1]
 #           int_flux = flux_cube.integral_flux_image(energy_bin)#

#        npred = int_flux * exposure * solid_angle
#    elif method = 'method2':
 #       data = exposure_cube.flux(energ, lon, lat)
 #       dnpred_denergy = SpectralCube()
 #       npred = dnpred_denergy.integral_flux_image()#

#    npred_cube = GammaSpectralCube(data=npred, )


def compute_npred_cube(flux_hdu_list, exposure_hdu_list, desired_energy=None,
                       convolve='Yes', max_convolution_offset=5):
    """ Computes predicted counts cube from model flux cube.

    Counts cube will be at energies in the exposure_hdu unless specified at a
    desired energy, for which a counts map at this energy is returned

    Parameters
    ----------
    flux_hdu_list : [`astropy.io.fits.ImageHDU`, `astropy.io.fits.TableHDU`]
        List of HDUs: Data cube (2 or 3 dimensional), Data table containing energies of slices
    exposure_hdu : [`astropy.io.fits.ImageHDU`, `astropy.io.fits.TableHDU`]
        List of HDUs: Data cube (2 or 3 dimensional), Data table containing energies of slices
    desired_energy : float (optional)
        Energy at which to return counts map
    convolve : bool
        Specify whether to convolve the counts cube with the Fermi/LAT PSF
    max_convolution_offset : float (if convolve = 'Yes')
        Size of convolution Kernel

    Returns
    -------
    npred_cube : `astropy.io.fits.ImageHDU`
        Data cube (2 or 3 dimensional) of predicted counts.
        If energy is specified, returns predicted counts image at given energy.
        Otherwise, returns predicted counts cube at energies of the exposure cube.

    Notes
    -----
    Returns the cube as a 3-dimensional array rather than a hdu object.
    """
    # TODO: Currently the smoothing option does not work, and hasn't been fixed
    # as it wasn't needed at the time...
    exposure_rep = _reproject_cube(exposure_hdu_list, flux_hdu_list, smooth=False)
    if desired_energy != None:
        flux_hdus = _interp_flux(flux_hdu_list, desired_energy)
        flux = flux_hdus[0].data
        exposure_hdus = _interp_exposure(exposure_rep, desired_energy)
        exposure = exposure_hdus[0].data
    else:
        flux_hdus = flux_hdu_list
        flux = flux_hdus[0].data
        exposure_hdus = exposure_rep
        exposure = exposure_hdus[0].data
    flux *= exposure
    if convolve == 'Yes':
        energies = flux_hdu_list[1].data['Energy']
        resolution = exposure_rep[0].header['CDELT2']
        indices = np.arange(len(energies))
        counts = np.zeros_like(flux)
        if len(flux.shape) == 3:
            for index in indices:
                cube_layer = flux[index - 1]
                eband_min = energies[index - 1]
                eband_max = energies[index]
                counts[index] = _correlate_fermi_psf(cube_layer, max_convolution_offset,
                                                     resolution, energy=np.sqrt(eband_min * eband_max))
                                                     # Is this OK re energy??
        elif len(flux.shape) == 2:
            if desired_energy != None:
                energy = desired_energy
            else:
                energy = flux_hdu_list[1].data['Energy']
            counts = _correlate_fermi_psf(flux, max_convolution_offset,
                                          resolution, energy)
        return counts
    else:
        counts = flux
        return counts


class GammaSpectralCube(object):
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

        self._interpolate_cache = None

    @property
    def _interpolate(self):
        if self._interpolate_cache == None:
            # Initialise the interpolator
            # This doesn't do any computations ... I'm not sure if it allocates extra arrays.
            from scipy.interpolate import RegularGridInterpolator
            points = list(map(np.arange, self.data.shape))
            self._interpolate_cache = RegularGridInterpolator(points, self.data.value,
                                                              fill_value=None, bounds_error=False)

        return self._interpolate_cache

    @staticmethod
    def read_hdu(hdu_list):
        """Read spectral cube from HDU.

        Parameters
        ----------
        object_hdu : `astropy.io.fits.ImageHDU`
            Image HDU object to be read
        energy_table_hdu : `astropy.io.fits.TableHDU`
            Table HDU object giving energies of each slice
            of the Image HDU object_hdu

        Returns
        -------
        spectral_cube : `GammaSpectralCube`
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
        return GammaSpectralCube(data, wcs, energy)

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
        x, y, _ = self.wcs.wcs_world2pix(lon, lat, 0, 0)

        z = self.energy_axis.world2pix(energy)

        shape = (x * y * z).shape
        x = x * np.ones(shape)
        y = y * np.ones(shape)
        z = z * np.ones(shape)

        if combine == True:
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
        lon, lat, _ = self.wcs.wcs_pix2world(x, y, 0, 0)
        energy = self.energy_axis.pix2world(z)

        lon = Quantity(lon, 'deg')
        lat = Quantity(lat, 'deg')
        energy = Quantity(energy, self.energy.unit)

        return lon, lat, energy

    @property
    def spatial_coordinates(self):
        """TODO: document.
        """
        n_lon = self.data.shape[2]
        n_lat = self.data.shape[1]
        i_lat, i_lon = np.indices((n_lat, n_lon))
        lon, lat, _ = self.pix2world(0, i_lat, i_lon)
        return lon, lat

    @property
    def solid_angle(self):
        pass

    def flux(self, lon, lat, energy):
        """Differential flux (linear interpolation).

        Parameters
        ----------
        lon : `~astropy.units.Quantity` or `~astropy.coordinates.Angle`
            Longitude
        lat : `~astropy.units.Quantity` or `~astropy.coordinates.Angle`
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
        values.reshape(shape)

        return Quantity(values, '1 / (cm2 MeV s sr)')

    def spectral_index(self, lon, lat, energy, dz=1e-3):
        """Power law spectral index.

        A forward finite difference method with step ``dz`` is used along
        the ``z = log10(energy)`` axis.

        Parameters
        ----------
        lon : `~astropy.units.Quantity` or `~astropy.coordinates.Angle`
            Longitude
        lat : `~astropy.units.Quantity` or `~astropy.coordinates.Angle`
            Latitude
        energy : `~astropy.units.Quantity`
            Energy

        Returns
        -------
        spectral_index : `~astropy.units.Quantity`
            Spectral index
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
            Integral flux image (1 / (m^2 s))
        """
        if isinstance(energy_bins, int):
            energy_bins = energy_bounds_equal_log_spacing(energy_band, energy_bins)

        energy1 = energy_bins[:-1]
        energy2 = energy_bins[1:]

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

        # TODO: check units ... set correctly if not OK.
        header = self.wcs.sub(['longitude', 'latitude']).to_header()
        hdu = fits.ImageHDU(data=integral_flux, header=header, name='integral_flux')

        return hdu

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

