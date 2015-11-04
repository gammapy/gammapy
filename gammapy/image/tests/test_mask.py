# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.wcs import WCS
from astropy.coordinates import Angle, SkyCoord
from astropy.io import fits
from ..mask import ExclusionMask
from ...datasets import gammapy_extra
from ...utils.testing import requires_data

@requires_data('gammapy-extra')
class TestSkyCircle:
    def setup(self):
        self.testfile = gammapy_extra.filename('test_datasets/spectrum/'
                                               'dummy_exclusion.fits')
        hdu = fits.open(self.testfile)[0]
        self.exclusion_mask = ExclusionMask.from_hdu(hdu)
        
    def test_distance_image(self):
        pass

