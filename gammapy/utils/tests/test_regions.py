# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Here we test the functions in `gammapy.utils.regions`.

We can also add tests for specific functionality and behaviour
in https://astropy-regions.readthedocs.io that we rely on in Gammapy.
That package is still work in progress and not fully developed and
stable, so need to establish a bit what works and what doesn't.
"""
from numpy.testing import assert_allclose, assert_equal
import astropy.units as u
import regions
from astropy.coordinates import SkyCoord
from gammapy.utils.regions import (
    SphericalCircleSkyRegion,
    make_region,
    compound_region_center
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


def test_compound_region_center():
    region_1 = make_region("galactic;circle(1,1,0.1)")
    region_2 = make_region("galactic;circle(-1,1,0.1)")
    region_3 = make_region("galactic;circle(1,-1,0.1)")
    region_4 = make_region("galactic;circle(-1,-1,0.1)")

    for region in [region_2, region_3, region_4]:
        region_1 = region_1.union(region)

    center = compound_region_center(region_1)

    assert_allclose(center.galactic.l.wrap_at("180d"), 0 * u.deg, atol=1e-6)
    assert_allclose(center.galactic.b, 0 * u.deg, atol=1e-6)


def test_compound_region_center_single():
    region = make_region("galactic;circle(1,1,0.1)")
    center = compound_region_center(region)

    assert_allclose(center.galactic.l.wrap_at("180d"), 1 * u.deg, atol=1e-6)
    assert_allclose(center.galactic.b, 1 * u.deg, atol=1e-6)


class TestSphericalCircleSkyRegion:
    def setup(self):
        self.region = SphericalCircleSkyRegion(
            center=SkyCoord(10 * u.deg, 20 * u.deg), radius=10 * u.deg
        )

    def test_contains(self):
        coord = SkyCoord([20.1, 22] * u.deg, 20 * u.deg)
        mask = self.region.contains(coord)
        assert_equal(mask, [True, False])
