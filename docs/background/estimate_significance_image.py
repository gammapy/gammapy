# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Estimate a Poisson significance image for a given counts image.

Here's a high-level description ... see the code for details:
* The background is estimated from a ring around each pixel.
* The significance is computed using the Li & Ma formula
  for the counts within a circle around each pixel.
* An iterative scheme is used to define an exclusion mask of
  pixels that shouldn't be used for background estimation.
"""
__doctest_skip__ = ['*']
import numpy as np
from astropy.io import fits
from gammapy.datasets import FermiGalacticCenter
from gammapy.image import threshold, binary_dilation_circle
from gammapy.background import Maps, ring_correlate_off_maps

counts = FermiGalacticCenter.counts()
binsz = counts.header['CDELT2']
maps = Maps([counts], theta=0.3)

for iteration in range(3):
    print('Iteration: {0}'.format(iteration))
    ring_correlate_off_maps(maps, r_in=0.5, r_out=0.8)
    significance = maps.significance.data
    #exclusion = threshold(significance, threshold=5)
    exclusion = np.where(significance > 4, 0, 1).astype(int)
    exclusion = binary_dilation_circle(exclusion, radius=0.4 * binsz)
    maps['exclusion'].data = exclusion.astype(np.uint8)

maps.make_derived_maps()
output_file = 'fermi_all_gc.fits'
print('Writing file: {0}'.format(output_file))
maps.writeto(output_file, clobber=True)
