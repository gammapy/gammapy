# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Cube analysis utility functions.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.io.fits import ImageHDU
from astropy.units import Quantity
from astropy.coordinates import Angle
from ..image import SkyImage
from .core import SkyCube
from ..spectrum import LogEnergyAxis

__all__ = [
    'compute_npred_cube',
    'convolve_cube',
    'cube_to_spec',
]


def compute_npred_cube(flux_cube, exposure_cube, energy_bins,
                       integral_resolution=10):
    """Computes predicted counts cube in energy bins.

    Parameters
    ----------
    flux_cube : `SkyCube`
        Differential flux cube.
    exposure_cube : `SkyCube`
        Instrument exposure cube.
    energy_bins : `~gammapy.utils.energy.EnergyBounds`
        An array of Quantities specifying the edges of the energy band
        required for the predicted counts cube.
    integral_resolution : int (optional)
        Number of integration steps in energy bin when computing integral flux.

    Returns
    -------
    npred_cube : `SkyCube`
        Predicted counts cube in energy bins.
    """
    if flux_cube.data.shape[1:] != exposure_cube.data.shape[1:]:
        raise ValueError('flux_cube and exposure cube must have the same shape!\n'
                         'flux_cube: {0}\nexposure_cube: {1}'
                         ''.format(flux_cube.data.shape[1:], exposure_cube.data.shape[1:]))

    energy_centers = energy_bins.log_centers
    wcs = exposure_cube.wcs
    coordinates = exposure_cube.sky_image_ref.coordinates()
    solid_angle = exposure_cube.sky_image_ref.solid_angle()

    npred_cube = np.zeros((len(energy_bins) - 1,
                           exposure_cube.data.shape[1], exposure_cube.data.shape[2]))
    for i in range(len(energy_bins) - 1):
        emin, emax = energy_bins[i:i + 2]
        int_flux = flux_cube.sky_image_integral(emin, emax, interpolation='linear', nbins=integral_resolution)
        int_flux = Quantity(int_flux.data, '1 / (cm2 s sr)')
        energy = energy_centers[i] * np.ones(coordinates.shape)
        exposure = Quantity(exposure_cube.lookup(coordinates, energy, interpolation='linear'), 'cm2 s')
        npred_image = int_flux * exposure * solid_angle
        npred_cube[i] = npred_image.to('')
    npred_cube = np.nan_to_num(npred_cube)

    energy_axis = LogEnergyAxis(energy_bins)
    npred_cube = SkyCube(data=npred_cube,
                         wcs=wcs,
                         energy_axis=energy_axis)
    return npred_cube


def convolve_cube(cube, psf, offset_max):
    """Convolves a predicted counts cube in energy bins with the an
    energy-dependent PSF.

    Parameters
    ----------
    cube : `SkyCube`
        Counts cube in energy bins.
    psf : `~gammapy.irf.EnergyDependentTablePSF`
        Energy dependent PSF.
    offset_max : `~astropy.units.Quantity`
        Maximum offset in degrees of the PSF convolution kernel from its center.

    Returns
    -------
    convolved_cube : `SkyCube`
        PSF convolved predicted counts cube in energy bins.
    """
    energy_edges = cube.energies(mode='edges')
    convolved_cube = SkyCube.empty_like(cube)

    data = []
    pixel_size = cube.sky_image_ref.wcs_pixel_scale()[0]

    for i, energy in enumerate(cube.energies()):
        image = cube.sky_image(energy)
        energy_band = energy_edges[i:i + 2]
        psf_at_energy = psf.table_psf_in_energy_band(energy_band)
        kernel = psf_at_energy.kernel(pixel_size, offset_max, normalize=True)
        convolved = image.convolve(kernel)
        data.append(convolved.data)

    convolved_cube.data = np.stack(data, axis=0)
    return convolved_cube


def cube_to_image(cube, slicepos=None):
    """Slice or project 3-dim cube into a 2-dim image.

    Parameters
    ----------
    cube : `~astropy.io.fits.ImageHDU`
        3-dim FITS cube
    slicepos : int or None, optional
        Slice position (None means to sum along the spectral axis)

    Returns
    -------
    image : `~astropy.io.fits.ImageHDU`
        2-dim FITS image
    """
    header = cube.header.copy()
    header['NAXIS'] = 2

    for key in ['NAXIS3', 'CRVAL3', 'CDELT3', 'CTYPE3', 'CRPIX3', 'CUNIT3']:
        if key in header:
            del header[key]

    if slicepos is None:
        data = cube.data.sum(0)
    else:
        data = cube.data[slicepos]
    return ImageHDU(data, header)


def cube_to_spec(cube, mask, weighting='none'):
    """Integrate spatial dimensions of a FITS cube to give a spectrum.

    TODO: give formulas.

    Parameters
    ----------
    cube : `~astropy.io.fits.ImageHDU`
        3-dim FITS cube
    mask : numpy.array
        2-dim mask array.
    weighting : {'none', 'solid_angle'}, optional
        Weighting factor to use.

    Returns
    -------
    spectrum : numpy.array
        Summed spectrum of pixels in the mask.
    """

    # TODO: clean up API and implementation and add test
    value = cube.dat
    image = SkyImage.read(cube)
    A = image.solid_angle()
    # Note that this is the correct way to get an average flux:

    spec = (value * A).sum(-1).sum(-1)
    return spec
