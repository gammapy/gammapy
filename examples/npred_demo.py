"""Test npred model image computation.
"""
from astropy.units import Quantity
from gammapy.datasets import FermiGalacticCenter
from gammapy.irf import EnergyDependentTablePSF
from gammapy.spectral_cube import GammaSpectralCube, compute_npred_cube, convolve_npred_cube

filenames = FermiGalacticCenter.filenames()
spectrum = GammaSpectralCube.read(filenames['diffuse_model'])
exposure = GammaSpectralCube.read(filenames['exposure_cube'])
exposure.spatial_coordinate_images

# Compute npred cube

energy_bounds = Quantity([10, 30, 100, 500], 'GeV')

npred_cube = compute_npred_cube(spectrum, exposure, energy_bounds)
print(npred_cube)


# PSF convolve the npred cube

psf = EnergyDependentTablePSF.read(filenames['psf'])

npred_cube_convolved = convolve_npred_cube(npred_cube, psf)
