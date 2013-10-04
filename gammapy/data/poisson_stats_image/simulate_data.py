# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Simulate test data (see README.md) with Sherpa"""
from __future__ import print_function, division
import numpy as np
import sherpa.astro.ui as sau 

# Define width of the source and the PSF
sigma_psf, sigma_source = 3, 4
# for relation of sigma and fwhm see
# http://cxc.harvard.edu/sherpa/ahelp/gauss2d.html
sigma_to_fwhm = np.sqrt(8 * np.log(2))  # ~ 2.35
sigma = np.sqrt(sigma_psf ** 2 + sigma_source ** 2)
fwhm = sigma_to_fwhm * sigma

# Seed the random number generator to make the output reproducible
np.random.seed(0)

sau.dataspace2d((200, 200))
sau.set_source('normgauss2d.source + const2d.background')
sau.set_par('source.xpos', 100)
sau.set_par('source.ypos', 100)
sau.set_par('source.ampl', 1e3)
sau.set_par('source.fwhm', fwhm)
sau.set_par('background.c0', 1)

sau.fake()
sau.save_model('model.fits.gz', clobber=True)
sau.save_data('counts.fits.gz', clobber=True)

sau.set_source('source')
sau.save_model('source.fits.gz', clobber=True)

sau.set_source('background')
sau.save_model('background.fits.gz', clobber=True)
