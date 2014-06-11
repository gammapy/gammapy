# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Compute Fermi PSF image for a given energy band.
"""
from astropy.coordinates import Angle
from astropy.units import Quantity
from astropy.io import fits
from gammapy.datasets import FermiGalacticCenter
from gammapy.irf import EnergyDependentTablePSF

# Parameters
filename = FermiGalacticCenter.filenames()['psf']
pixel_size = Angle(0.1, 'deg')
offset_max = Angle(2, 'deg')
energy = Quantity(10, 'GeV')
energy_band = Quantity([10, 500], 'GeV')
outfile = 'fermi_psf_image.fits'

# Compute PSF image
fermi_psf = EnergyDependentTablePSF.read(filename)
#psf = fermi_psf.table_psf_at_energy(energy=energy)
psf = fermi_psf.table_psf_in_energy_band(energy_band=energy_band, spectral_index=2.5)
psf.normalize()
kernel = psf.kernel(pixel_size=pixel_size, offset_max=offset_max)
kernel_image = kernel.value

kernel_image_integral = kernel_image.sum() * pixel_size.to('radian').value ** 2
print('Kernel image integral: {0}'.format(kernel_image_integral))
print('shape: {0}'.format(kernel_image.shape))

#import IPython; IPython.embed()
# Print some info and save to FITS file
#print(fermi_psf.info())

print(psf.info())
print('Writing {0}'.format(outfile))
fits.writeto(outfile, data=kernel_image, clobber=True)
