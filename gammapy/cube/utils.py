# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = ['compute_npred_cube', 'convolve_cube']


def compute_npred_cube(flux_cube, exposure_cube, energy_bins,
                       integral_resolution=10):
    """Computes predicted counts cube in energy bins.

    Parameters
    ----------
    flux_cube : `SpectralCube`
        Differential flux cube.
    exposure_cube : `SpectralCube`
        Instrument exposure cube.
    energy_bins : `~gammapy.utils.energy.EnergyBounds`
        An array of Quantities specifying the edges of the energy band
        required for the predicted counts cube.
    integral_resolution : int (optional)
        Number of integration steps in energy bin when computing integral flux.

    Returns
    -------
    npred_cube : `SpectralCube`
        Predicted counts cube in energy bins.
    """
    if flux_cube.data.shape[1:] != exposure_cube.data.shape[1:]:
        raise ValueError('flux_cube and exposure cube must have the same shape!\n'
                         'flux_cube: {0}\nexposure_cube: {1}'
                         ''.format(flux_cube.data.shape[1:], exposure_cube.data.shape[1:]))

    energy_centers = energy_bins.log_centers
    wcs = exposure_cube.wcs
    lon, lat = exposure_cube.spatial_coordinate_images
    solid_angle = exposure_cube.solid_angle_image
    npred_cube = np.zeros((len(energy_bins) - 1,
                           exposure_cube.data.shape[1], exposure_cube.data.shape[2]))
    for i in range(len(energy_bins) - 1):
        energy_bound = energy_bins[i:i + 2]
        int_flux = flux_cube.integral_flux_image(energy_bound, integral_resolution)
        int_flux = Quantity(int_flux.data, '1 / (cm2 s sr)')
        exposure = Quantity(exposure_cube.flux(lon, lat,
                                               energy_centers[i]).value, 'cm2 s')
        npred_image = int_flux * exposure * solid_angle
        npred_cube[i] = npred_image.to('')
    npred_cube = np.nan_to_num(npred_cube)

    npred_cube = SpectralCube(data=npred_cube,
                              wcs=wcs,
                              energy=energy_bins)
    return npred_cube


def convolve_cube(cube, psf, offset_max):
    """Convolves a predicted counts cube in energy bins with the an
    energy-dependent PSF.

    Parameters
    ----------
    cube : `SpectralCube`
        Counts cube in energy bins.
    psf : `~gammapy.irf.EnergyDependentTablePSF`
        Energy dependent PSF.
    offset_max : `~astropy.units.Quantity`
        Maximum offset in degrees of the PSF convolution kernel from its center.

    Returns
    -------
    convolved_cube : `SpectralCube`
        PSF convolved predicted counts cube in energy bins.
    """
    from scipy.ndimage import convolve
    energy = cube.energy
    indices = np.arange(len(energy) - 1)
    convolved_cube = np.zeros_like(cube.data)
    pixel_size = Angle(np.abs(cube.wcs.wcs.cdelt[0]), 'deg')

    for i in indices:
        energy_band = energy[i:i + 2]
        psf_at_energy = psf.table_psf_in_energy_band(energy_band)
        kernel_image = psf_at_energy.kernel(pixel_size, offset_max, normalize=True)
        convolved_cube[i] = convolve(cube.data[i], kernel_image,
                                     mode='mirror')
    convolved_cube = SpectralCube(data=convolved_cube, wcs=cube.wcs,
                                  energy=cube.energy)
    return convolved_cube