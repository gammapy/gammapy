# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Compute results with Sherpa"""
from __future__ import print_function, division
# __doctest_skip__
__doctest_skip__ = ['*']
import numpy as np
import sherpa.astro.ui as sau

sau.load_data('counts.fits.gz')
sau.set_source('normgauss2d.source + const2d.background')
sau.set_stat('cstat')
# Ask for high-precision results
sau.set_method_opt('ftol', 1e-20)
sau.set_covar_opt('eps', 1e-20)

# Set start parameters close to simulation values to make the fit converge
sau.set_par('source.xpos', 101)
sau.set_par('source.ypos', 101)
sau.set_par('source.ampl', 1.1e3)
sau.set_par('source.fwhm', 10)
sau.set_par('background.c0', 1.1)

# Run fit and covariance estimation
# Results are automatically printed to the screen
sau.fit()
sau.covar()

# Sherpa uses fwhm instead of sigma as extension parameter ... need to convert
# http://cxc.harvard.edu/sherpa/ahelp/gauss2d.html
fwhm_to_sigma = 1. / np.sqrt(8 * np.log(2))
cov = sau.get_covar_results()
sigma = fwhm_to_sigma * cov.parvals[0]
sigma_err = fwhm_to_sigma * cov.parmaxes[0]
print('sigma: {0} +- {1}'.format(sigma, sigma_err))

# Compute correlation coefficient for sigma and norm
c = cov.extra_output
c_norm = c[3, 3]
c_sigma = fwhm_to_sigma ** 2 * c[0, 0]
c_norm_sigma = fwhm_to_sigma * c[0, 3]
corr_norm_sigma = c_norm_sigma / np.sqrt(c_norm * c_sigma)
print('corr_norm_sigma: {0}'.format(corr_norm_sigma))

# Save model excess image
sau.save_model('model_sherpa.fits.gz', clobber=True)

# Compute TS
L1 = sau.calc_stat()
sau.set_source('const2d.background')
sau.fit()
L0 = sau.calc_stat()
TS = 2 * (L0 - L1)
print('TS: {:.5f}'.format(TS))

