"""Runs commands to produce convolved predicted counts map in current directory
"""
import numpy as np
from astropy.coordinates import Angle
from astropy.units import Quantity
from astropy.io import fits
from astropy.wcs import WCS
from gammapy.data import (SpectralCube,
                          compute_npred_cube,
                          convolve_cube)
from gammapy.datasets import FermiVelaRegion
from gammapy.irf import EnergyDependentTablePSF

__all__ = ['prepare_images']


def prepare_images():
    # Reads in data
    background_file = FermiVelaRegion.filenames()['diffuse_model']
    exposure_file = FermiVelaRegion.filenames()['exposure_cube']
    counts_file = FermiVelaRegion.filenames()['counts_cube']
    background_model = SpectralCube.read(background_file)
    exposure_cube = SpectralCube.read(exposure_file)
    # Add correct units
    exposure_cube.data = Quantity(exposure_cube.data, '1/(cm2 deg2 s GeV)')
    # Re-project background cube
    repro_bg_cube = background_model.reproject_to(exposure_cube)
    # Define energy band required for output
    energies = Quantity([10, 500], 'GeV')

    # Compute the predicted counts cube
    npred_cube = compute_npred_cube(repro_bg_cube, exposure_cube, energies)

    # Convolve with Energy-dependent Fermi LAT PSF

    psf = EnergyDependentTablePSF.read(FermiVelaRegion.filenames()['psf'])
    convolved_npred_cube = convolve_cube(npred_cube, psf,
                                         offset_max=Angle(3, 'deg'))

    # Counts data
    counts_data = fits.open(counts_file)[0].data
    counts_wcs = WCS(fits.open(counts_file)[0].header)
    counts_cube = SpectralCube(data=Quantity(counts_data, ''),
                               wcs=counts_wcs,
                               energy=energies)
    counts_cube = counts_cube.reproject_to(npred_cube)

    counts = counts_cube.data[0]
    model = convolved_npred_cube.data[0]

    # Load Fermi tools gtmodel background-only result

    gtmodel = fits.open(FermiVelaRegion.filenames()['background_image'])[0].data.astype(float)
    # Ratio for the two background images
    ratio = np.nan_to_num(model / gtmodel)

    return model, gtmodel, ratio, counts
