# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.wcs import WCS
from astropy.coordinates import Angle, SkyCoord
from ...image import make_empty_image
from ..circle import SkyCircleRegion


class TestSkyCircle:

    def setup(self):
        hdu = make_empty_image(nxpix=200, nypix=100)
        self.wcs = WCS(hdu.header)

    def test_sky_to_pix(self):
        pos1 = SkyCoord(2, 1, unit='deg', frame='galactic')
        radius = Angle(1, 'deg')
        sky = SkyCircleRegion(pos=pos1, radius=radius)
        pix = sky.to_pixel(self.wcs)
        print(sky.__dict__)
        print(pix.__dict__)
        1/0

        # TODO: assert on pix pos and radius
