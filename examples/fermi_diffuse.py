# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Compute integral flux in an energy band for the Fermi diffuse model.
"""
import numpy as np
from astropy.units import Quantity
from gammapy.datasets import get_fermi_diffuse_background_model
from gammapy.spectral_cube import GammaSpectralCube

filename = get_fermi_diffuse_background_model()
cube = GammaSpectralCube.read(filename)
print(cube)

energy_band = Quantity([10, 30], 'GeV')
image = cube.integral_flux_image(energy_band)
print(image.sum())
