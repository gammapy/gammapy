"""Runs commands to produce convolved predicted counts map in current directory."""
import numpy as np
from astropy.coordinates import Angle
from astropy.units import Quantity
from astropy.io import fits
from astropy.wcs import WCS
from gammapy.cube import SkyCube, compute_npred_cube
from gammapy.datasets import FermiVelaRegion
from gammapy.irf import EnergyDependentTablePSF
from gammapy.utils.energy import EnergyBounds

__all__ = ['prepare_images']


def prepare_images():
    # Read in data
    fermi_vela = FermiVelaRegion()
    background_file = FermiVelaRegion.filenames()['diffuse_model']
    exposure_file = FermiVelaRegion.filenames()['exposure_cube']
    counts_file = FermiVelaRegion.filenames()['counts_cube']
    background_model = SkyCube.read(background_file, format='fermi-background')
    exposure_cube = SkyCube.read(exposure_file, format='fermi-exposure')

    # Re-project background cube
    repro_bg_cube = background_model.reproject(exposure_cube)

    # Define energy band required for output
    energies = EnergyBounds([10, 500], 'GeV')

    # Compute the predicted counts cube
    npred_cube = compute_npred_cube(repro_bg_cube, exposure_cube, energies,
                                    integral_resolution=5)

    # Convolve with Energy-dependent Fermi LAT PSF
    psf = EnergyDependentTablePSF.read(FermiVelaRegion.filenames()['psf'])
    kernels = psf.kernels(npred_cube)
    convolved_npred_cube = npred_cube.convolve(kernels, mode='reflect')

    # Counts data
    counts_cube = SkyCube.read(counts_file, format='fermi-counts')
    counts_cube = counts_cube.reproject(npred_cube)

    counts = counts_cube.data[0]
    model = convolved_npred_cube.data[0]

    # Load Fermi tools gtmodel background-only result
    gtmodel = fits.open(FermiVelaRegion.filenames()['background_image'])[0].data.astype(float)

    # Ratio for the two background images
    ratio = np.nan_to_num(model / gtmodel)

    # Header is required for plotting, so returned here
    wcs = npred_cube.wcs
    header = wcs.to_header()
    return model, gtmodel, ratio, counts, header
