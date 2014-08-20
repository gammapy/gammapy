#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Given a catalog of sources, simulate a flux image. 
"""

from astropy.io import fits
from image.utils import empty_image
from image.simulate import _to_image_bbox as to_image

catalog = fits.open('test_catalog.fits')[1].data
image = empty_image(nxpix=600, nypix=600, binsz=0.02, xref=0, yref=0, dtype='float64')
to_image(catalog, image)
image.writetofits('test_image.fits', clobber=True)