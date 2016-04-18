"""Test npred model image computation.
"""
from astropy.coordinates import Angle
from gammapy.datasets import FermiGalacticCenter
from gammapy.utils.energy import EnergyBounds
from gammapy.irf import EnergyDependentTablePSF
from gammapy.cube import SkyCube, compute_npred_cube, convolve_cube

filenames = FermiGalacticCenter.filenames()
flux_cube = SkyCube.read(filenames['diffuse_model'])
exposure_cube = SkyCube.read(filenames['exposure_cube'])
psf = EnergyDependentTablePSF.read(filenames['psf'])

flux_cube = flux_cube.reproject_to(exposure_cube)

energy_bounds = EnergyBounds([10, 30, 100, 500], 'GeV')
npred_cube = compute_npred_cube(flux_cube, exposure_cube, energy_bounds)

offset_max = Angle(1, 'deg')
npred_cube_convolved = convolve_cube(npred_cube, psf, offset_max)
