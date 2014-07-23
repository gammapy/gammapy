"""Test npred model image computation.
"""
from astropy.units import Quantity
from gammapy.datasets import FermiGalacticCenter
from gammapy.irf import EnergyDependentTablePSF
from gammapy.spectral_cube import GammaSpectralCube, compute_npred_cube

filenames = FermiGalacticCenter.filenames()
spectrum = GammaSpectralCube.read(filenames['diffuse_model'])
exposure = GammaSpectralCube.read(filenames['exposure_cube'])
psf = EnergyDependentTablePSF.read(filenames['psf'])

energy_bounds = Quantity([10, 30, 100, 500], 'GeV')

npred = compute_npred_cube(spectrum, exposure, energy_bounds)

print(npred)
