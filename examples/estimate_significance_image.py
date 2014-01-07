# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Estimate a Poisson significance image for a given counts image.

Here's a high-level description ... see the code for details:
* The background is estimated from a ring around each pixel.
* The significance is computed using the Li & Ma formula
  for the counts within a circle around each pixel.
* An iterative scheme is used to define an exclusion mask of
  pixels that shouldn't be used for background estimation.
"""
import numpy as np
from astropy.io import fits
from gammapy.image import threshold, dilate
from gammapy.background import Maps, ring_correlate_off_maps

input_file = 'fermi_counts_gc.fits'
print('Reading file: {0}'.format(input_file))
hdu_list = fits.open(input_file)
binsz = hdu_list[-1].header['CDELT2']
maps = Maps(hdu_list, theta=0.3)

for iteration in range(3):
    print('Iteration: {0}'.format(iteration))
    ring_correlate_off_maps(maps, r_in=0.5, r_out=0.8)
    significance = maps.make_significance().data
    #exclusion = threshold(significance, threshold=5)
    exclusion = np.where(significance > 4, 0, 1).astype(int)
    exclusion = dilate(exclusion, radius=0.4 * binsz)
    maps['exclusion'].data = exclusion

maps.make_derived_maps()
output_file = 'fermi_all_gc.fits'
print('Writing file: {0}'.format(output_file))
maps.writeto(output_file, clobber=True)
