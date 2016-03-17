# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.tests.helper import pytest
from astropy.wcs import WCS
from astropy.coordinates import Angle, SkyCoord
from ...image import SkyMap
from ..circle import SkyCircleRegion, PixCircleRegion
from numpy.testing import assert_allclose


@pytest.fixture
def wcs():
    """Example WCS object for testing."""
    return SkyMap.empty(nxpix=201, nypix=101, binsz=0.1).wcs

def test_sky_to_pix(wcs):
    pos = SkyCoord(2, 1, unit='deg', frame='galactic')
    radius = Angle(1, 'deg')
    sky = SkyCircleRegion(pos=pos, radius=radius)

    pix = sky.to_pixel(wcs)

    assert_allclose(pix.radius, 10)
    assert_allclose(pix.pos[0], 81)
    assert_allclose(pix.pos[1], 61)


def test_sky_to_pix2():
    hdu = SkyMap.empty(nxpix=801, nypix=601, binsz=0.01,
                       coordsys='CEL', xref=83.2, yref=22.7).to_image_hdu()

    pos = SkyCoord(182.2, -5.75, unit='deg', frame='galactic')
    radius = Angle(0.4, 'deg')
    sky = SkyCircleRegion(pos=pos, radius=radius)
    pix = sky.to_pixel(WCS(hdu.header))

    assert_allclose(pix.radius, 40)


def test_pix_to_sky(wcs):
    pix = PixCircleRegion(pos=(61, 31), radius=5)

    sky = pix.to_sky(wcs, frame='galactic')

    assert_allclose(sky.radius.value, 0.5)
    assert_allclose(sky.pos.l.value, 4)
    assert_allclose(sky.pos.b.value, -2)


def test_sky_to_pix_to_sky(wcs):
    pos1 = SkyCoord(5, 3, unit='deg', frame='galactic')
    radius = Angle(1.5, 'deg')
    sky = SkyCircleRegion(pos=pos1, radius=radius)
    pix = sky.to_pixel(wcs)
    sky2 = pix.to_sky(wcs)

    assert_allclose(sky.pos.l, sky2.pos.l)
    assert_allclose(sky.pos.b, sky2.pos.b)
    assert_allclose(sky.radius, sky2.radius)


def test_area():
    pos = SkyCoord(83.633083, 22.0145, unit='deg')
    rad = Angle('1.3451 deg')
    reg = SkyCircleRegion(pos, rad)

    # small angle approximation, method: spherical cap area
    desired = (rad ** 2 * np.pi).to('steradian')
    assert_allclose(reg.area, desired, rtol=1e-3)
