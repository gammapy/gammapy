# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Simulate test data (see README.md) with Sherpa"""
from __future__ import print_function, division
import json

import numpy as np

from astropy.modeling.models import Gaussian2D, Const2D
from astropy.io import fits
from astropy.wcs import WCS

from gammapy.utils.random import get_random_state

# Define width of the source and the PSF
sigma_psf, sigma_source = 3, 4
sigma = np.sqrt(sigma_psf ** 2 + sigma_source ** 2)
amplitude = 1E3 / (2 * np.pi * sigma ** 2)

source = Gaussian2D(amplitude, 99, 99, sigma, sigma)
background = Const2D(1)
model = source + background


# Define data shape
shape = (200, 200)
y, x = np.indices(shape)

# Create a new WCS object
w = WCS(naxis=2)

# Set up an Galactic projection
w.wcs.crpix = [99, 99]
w.wcs.cdelt = np.array([0.02, 0.02])
w.wcs.crval = [0, 0]
w.wcs.ctype = ['GLON-CAR', 'GLAT-CAR']

# Fake data
random_state = get_random_state(0)
data = random_state.poisson(model(x, y))

# Save data
header = w.to_header()

hdu = fits.PrimaryHDU(data=data.astype('int16'), header=header)
hdu.writeto('counts.fits.gz', clobber=True)

hdu = fits.PrimaryHDU(data=model(x, y).astype('float32'), header=header)
hdu.writeto('model.fits.gz', clobber=True)

hdu = fits.PrimaryHDU(data=background(x, y).astype('int16'), header=header)
hdu.writeto('background.fits.gz', clobber=True)

hdu = fits.PrimaryHDU(data=source(x, y).astype('float32'), header=header)
hdu.writeto('source.fits.gz', clobber=True)

exposure = 1E12 * np.ones(shape)
hdu = fits.PrimaryHDU(data=exposure.astype('float32'), header=header)
hdu.writeto('exposure.fits.gz', clobber=True)


# save psf info in json format
psf = {}
psf['psf1'] = {'ampl': 1, 'fwhm': sigma_psf * 2 * np.sqrt(2 * np.log(2))}
psf['psf2'] = {'ampl': 0, 'fwhm': 1E-5}
psf['psf3'] = {'ampl': 0, 'fwhm': 1E-5}

with open('psf.json', 'w') as f:
    json.dump(psf, f, indent=4)

