"""Test npred model image computation.
"""
from astropy.coordinates import Angle
from gammapy.datasets import FermiGalacticCenter
from gammapy.utils.energy import EnergyBounds
from gammapy.irf import EnergyDependentTablePSF
from gammapy.cube import SkyCube, compute_npred_cube

filenames = FermiGalacticCenter.filenames()
flux_cube = SkyCube.read(filenames['diffuse_model'], format='fermi-background')
exposure_cube = SkyCube.read(filenames['exposure_cube'], format='fermi-exposure')
psf = EnergyDependentTablePSF.read(filenames['psf'])

flux_cube = flux_cube.reproject(exposure_cube)

energy_bounds = EnergyBounds([10, 30, 100, 500], 'GeV')
npred_cube = compute_npred_cube(flux_cube, exposure_cube, energy_bounds)

# offset_max = Angle(1, 'deg')

# TODO: this is just a 3x3 kernel.
# Find option to make it larger to avoid cutoff.
kernels = psf.kernels(npred_cube)
npred_cube_convolved = npred_cube.convolve(kernels)
