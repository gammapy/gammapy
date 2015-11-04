# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.io import fits
from astropy.coordinates import SkyCoord, Angle
from ..reflected import find_reflected_regions
from ..mask import ExclusionMask
from ..circle import SkyCircleRegion
from ...datasets import gammapy_extra
from ...utils.testing import requires_data

@requires_data('gammapy-extra')
def test_find_reflected_regions():

    testfile = gammapy_extra.filename('test_datasets/spectrum/dummy_exclusion.fits')
    hdu = fits.open(testfile)[0]
    mask = ExclusionMask.from_hdu(hdu)
    pos = SkyCoord(80.2, 23.5, unit='deg', frame='icrs')
    radius = Angle(0.4, 'deg')
    region = SkyCircleRegion(pos=pos, radius=radius)
    center = SkyCoord(83.2, 22.7, unit='deg', frame='icrs')
    regions = find_reflected_regions(region, center, mask)

    assert len(regions) == 15
    
