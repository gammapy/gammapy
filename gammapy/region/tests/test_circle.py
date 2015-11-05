# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.wcs import WCS
from astropy.coordinates import Angle, SkyCoord
from ...image import make_empty_image
from ..circle import SkyCircleRegion, PixCircleRegion
from numpy.testing import assert_allclose

class TestSkyCircle:

    def setup(self):
        hdu = make_empty_image(nxpix=201, nypix=101)
        self.wcs = WCS(hdu.header)

    def test_sky_to_pix(self):
        pos = SkyCoord(2, 1, unit='deg', frame='galactic')
        radius = Angle(1, 'deg')
        sky = SkyCircleRegion(pos=pos, radius=radius)
        pix = sky.to_pixel(self.wcs)
        assert_allclose(pix.radius, 10)
        assert_allclose(pix.pos[0], 81)
        assert_allclose(pix.pos[1], 61)

    def test_sky_to_pix2(self):
        hdu = make_empty_image(nxpix=801, nypix=601, binsz=0.01, 
                               coordsys='CEL', xref=83.2, yref=22.7)
        
        pos = SkyCoord(182.2, -5.75, unit='deg', frame='galactic')
        radius = Angle(0.4, 'deg')
        sky = SkyCircleRegion(pos=pos, radius=radius)
        pix = sky.to_pixel(WCS(hdu.header))
        assert_allclose(pix.radius, 40)
        
    def test_pix_to_sky(self):
        pos = (61,31)
        radius = 5
        pix = PixCircleRegion(pos=pos, radius=radius)
        sky = pix.to_sky(self.wcs, frame='galactic')
        assert_allclose(sky.radius.value, 0.5)
        assert_allclose(sky.pos.l.value, 4)
        assert_allclose(sky.pos.b.value, -2)

    def test_sky_to_pix_to_sky(self):
        pos1 = SkyCoord(5, 3, unit='deg', frame='galactic')
        radius = Angle(1.5, 'deg')
        sky = SkyCircleRegion(pos=pos1, radius=radius)
        pix = sky.to_pixel(self.wcs)
        sky2 = pix.to_sky(self.wcs)
        assert_allclose(sky.pos.l, sky2.pos.l)
        assert_allclose(sky.pos.b, sky2.pos.b)
        assert_allclose(sky.radius, sky2.radius)
