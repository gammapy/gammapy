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
from astropy.coordinates import SkyCoord
from regions import CircleSkyRegion, EllipseSkyRegion, Regions
from gammapy.utils.regions import (
    SphericalCircleSkyRegion,
    compound_region_center,
    region_circle_to_ellipse,
    region_to_frame,
    regions_to_compound_region,
)


def test_compound_region_center():
    regions_ds9 = (
        "galactic;"
        "circle(1,1,0.1);"
        "circle(-1,1,0.1);"
        "circle(1,-1,0.1);"
        "circle(-1,-1,0.1);"
    )

    regions = Regions.parse(regions_ds9, format="ds9")

    region = regions_to_compound_region(regions)

    center = compound_region_center(region)

    assert_allclose(center.galactic.l.wrap_at("180d"), 0 * u.deg, atol=1e-6)
    assert_allclose(center.galactic.b, 0 * u.deg, atol=1e-6)


def test_compound_region_center_single():
    region = Regions.parse("galactic;circle(1,1,0.1)", format="ds9")[0]
    center = compound_region_center(region)

    assert_allclose(center.galactic.l.wrap_at("180d"), 1 * u.deg, atol=1e-6)
    assert_allclose(center.galactic.b, 1 * u.deg, atol=1e-6)


def test_compound_region_center_concentric():
    regions_ds9 = "galactic;" "circle(0,0,0.1);" "circle(0,0,0.2);"

    regions = Regions.parse(regions_ds9, format="ds9")

    region = regions_to_compound_region(regions)

    center = compound_region_center(region)

    assert_allclose(center.galactic.l.wrap_at("180d"), 0 * u.deg, atol=1e-6)
    assert_allclose(center.galactic.b, 0 * u.deg, atol=1e-6)


def test_compound_region_center_inomogeneous_frames():
    region_icrs = Regions.parse("icrs;circle(1,1,0.1)", format="ds9")[0]
    region_galactic = Regions.parse("galactic;circle(1,1,0.1)", format="ds9")[0]

    regions = Regions([region_icrs, region_galactic])

    region = regions_to_compound_region(regions)

    center = compound_region_center(region)

    assert_allclose(center.galactic.l.wrap_at("180d"), 28.01 * u.deg, atol=1e-2)
    assert_allclose(center.galactic.b, -37.46 * u.deg, atol=1e-2)


def test_spherical_circle_sky_region():
    region = SphericalCircleSkyRegion(
        center=SkyCoord(10 * u.deg, 20 * u.deg), radius=10 * u.deg
    )

    coord = SkyCoord([20.1, 22] * u.deg, 20 * u.deg)
    mask = region.contains(coord)
    assert_equal(mask, [True, False])


def test_region_to_frame():
    region = EllipseSkyRegion(
        center=SkyCoord(20, 17, unit="deg"),
        height=0.3 * u.deg,
        width=1.0 * u.deg,
        angle=30 * u.deg,
    )
    region_new = region_to_frame(region, "galactic")
    assert_allclose(region_new.angle, 20.946 * u.deg, rtol=1e-3)
    assert_allclose(region_new.center.l, region.center.galactic.l, rtol=1e-3)


def test_region_circle_to_ellipse():
    region = CircleSkyRegion(center=SkyCoord(20, 17, unit="deg"), radius=1.0 * u.deg)
    region_new = region_circle_to_ellipse(region)
    assert_allclose(region_new.height, region.radius, rtol=1e-3)
    assert_allclose(region_new.width, region.radius, rtol=1e-3)
    assert_allclose(region_new.angle, 0.0 * u.deg, rtol=1e-3)
