# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Here we test the functions in `gammapy.utils.regions`.

We can also add tests for specific functionality and behaviour
in https://astropy-regions.readthedocs.io that we rely on in Gammapy.
That package is still work in progress and not fully developed and
stable, so need to establish a bit what works and what doesn't.
"""
import pytest
from numpy.testing import assert_allclose, assert_equal
import astropy.units as u
import regions
from astropy.coordinates import SkyCoord
from gammapy.maps import WcsGeom
from gammapy.utils.regions import (
    SphericalCircleSkyRegion,
    make_pixel_region,
    make_region,
)


def test_make_region():
    reg = make_region("image;circle(10,20,3)")
    assert isinstance(reg, regions.CirclePixelRegion)
    assert reg.center.x == 9
    assert reg.center.y == 19
    assert reg.radius == 3

    reg = make_region("galactic;circle(10,20,3)")
    assert reg.center.l.deg == 10
    assert reg.center.b.deg == 20
    assert reg.radius.to_value("deg") == 3

    # Existing regions should pass through
    reg2 = make_region(reg)
    assert reg is reg2

    with pytest.raises(TypeError):
        make_pixel_region([reg])


def test_make_pixel_region():
    wcs = WcsGeom.create().wcs

    reg = make_pixel_region("image;circle(10,20,3)")
    assert isinstance(reg, regions.CirclePixelRegion)
    assert reg.center.x == 9
    assert reg.center.y == 19
    assert reg.radius == 3

    reg = make_pixel_region("galactic;circle(10,20,3)", wcs)
    assert isinstance(reg, regions.CirclePixelRegion)
    assert_allclose(reg.center.x, 570.9301128316974)
    assert_allclose(reg.center.y, 159.935542455567)
    assert_allclose(reg.radius, 6.061376992149382)

    with pytest.raises(ValueError):
        make_pixel_region("galactic;circle(10,20,3)")

    with pytest.raises(TypeError):
        make_pixel_region(99)


class TestSphericalCircleSkyRegion:
    def setup(self):
        self.region = SphericalCircleSkyRegion(
            center=SkyCoord(10 * u.deg, 20 * u.deg), radius=10 * u.deg
        )

    def test_contains(self):
        coord = SkyCoord([20.1, 22] * u.deg, 20 * u.deg)
        mask = self.region.contains(coord)
        assert_equal(mask, [True, False])
