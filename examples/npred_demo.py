"""Test npred model image computation.
"""
from astropy.units import Quantity
from gammapy.datasets import FermiGalacticCenter
from gammapy.irf import EnergyDependentTablePSF
from gammapy.spectral_cube import (GammaSpectralCube,
                                   compute_npred_cube,
                                   convolve_npred_cube
                                   )

filenames = FermiGalacticCenter.filenames()
spectral_cube = GammaSpectralCube.read(filenames['diffuse_model'])
exposure_cube = GammaSpectralCube.read(filenames['exposure_cube'])
psf = EnergyDependentTablePSF.read(filenames['psf'])

energy_bounds = Quantity([10, 30, 100, 500], 'GeV')

# Reproject spectral cube onto exposure cube

spectral_cube = spectral_cube.reproject_to(exposure_cube)

# Compute npred cube

npred_cube = compute_npred_cube(spectral_cube,
                                exposure_cube,
                                energy_bounds)
print(npred_cube)


# PSF convolve the npred cube

npred_cube_convolved = convolve_npred_cube(npred_cube, psf)
