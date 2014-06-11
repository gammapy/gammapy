# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Compute integral flux in an energy band for the Fermi diffuse model.
"""
from astropy.units import Quantity
from gammapy.datasets import FermiGalacticCenter

cube = FermiGalacticCenter.diffuse_model()
print(cube)

energy_band = Quantity([10, 50], 'GeV')
image = cube.integral_flux_image(energy_band, energy_bins=100)
image.writeto('fermi_diffuse_integral_flux_image.fits', clobber=True)

# Some checks
surface_brightness = Quantity(image.data.mean(), 'cm^-2 s^-1 sr^-1')
print('Mean surface brightness in image: {0}'.format(surface_brightness))
