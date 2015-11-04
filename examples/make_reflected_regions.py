# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.io import fits
from astropy.coordinates import SkyCoord, Angle
from gammapy.region import find_reflected_regions, ExclusionMask, SkyCircleRegion
from gammapy.datasets import gammapy_extra
from gammapy.utils.testing import requires_data

testfile = gammapy_extra.filename('test_datasets/spectrum/dummy_exclusion.fits')
hdu = fits.open(testfile)[0]
mask = ExclusionMask.from_hdu(hdu)
pos = SkyCoord(80.2, 23.5, unit='deg', frame='icrs')
radius = Angle(0.1, 'deg')
region = SkyCircleRegion(pos=pos, radius=radius)
center = SkyCoord(83.2, 22.7, unit='deg', frame='icrs')
regions = find_reflected_regions(region, center, mask)
regions.write('test.reg')

