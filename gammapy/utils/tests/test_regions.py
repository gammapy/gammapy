# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Here we test the functions in `gammapy.utils.regions`.

We can also add tests for specific functionality and behaviour
in https://astropy-regions.readthedocs.io that we rely on in Gammapy.
That package is still work in progress and not fully developed and
stable, so need to establish a bit what works and what doesn't.
"""

import numpy as np
from numpy.testing import assert_allclose, assert_equal
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from regions import (
    CircleSkyRegion,
    EllipseSkyRegion,
    RegionBoundingBox,
    Regions,
    RegionMeta,
    RegionVisual,
    PixCoord,
)
from astropy.wcs import WCS

from gammapy.utils.regions import (
    CirclePixelRegionArray,
    CircleSkyRegionArray,
    SphericalCircleSkyRegion,
    compound_region_center,
    region_circle_to_ellipse,
    region_to_frame,
    regions_to_compound_region,
    get_centroid,
    PolygonPointsSkyRegion,
    PolygonPointsPixelRegion,
    extract_bright_star_regions,
    make_grid_rectangle_sky_regions,
)

from gammapy.maps import WcsGeom, MapAxis, RegionGeom, RegionNDMap
from gammapy.utils.testing import requires_data
from gammapy.data import EventList


def test_compound_region_center():
    regions_ds9 = (
        "galactic;circle(1,1,0.1);circle(-1,1,0.1);circle(1,-1,0.1);circle(-1,-1,0.1);"
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
    regions_ds9 = "galactic;circle(0,0,0.1);circle(0,0,0.2);"

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


def test_get_centroid():
    vertices = SkyCoord(
        [
            (0, 0),
            (0, 1),
            (1, 1),
            (1, 0),
        ],
        unit="deg",
    )

    expected_centroid = SkyCoord(0.5 * u.deg, 0.5 * u.deg)

    calculated_centroid = get_centroid(vertices)

    assert_allclose(calculated_centroid.ra.deg, expected_centroid.ra.deg, rtol=1e-7)
    assert_allclose(calculated_centroid.dec.deg, expected_centroid.dec.deg, rtol=1e-7)


def test_polygon_points_sky_region_init():
    vertices = SkyCoord([(0, 0), (0, 1), (1, 1), (1, 0)], unit="deg")
    region = PolygonPointsSkyRegion(vertices)
    assert np.all(region.vertices == vertices)
    assert isinstance(region.meta, RegionMeta)
    assert isinstance(region.visual, RegionVisual)
    expected_centroid = SkyCoord(0.5 * u.deg, 0.5 * u.deg)
    assert_allclose(region.center.ra.deg, expected_centroid.ra.deg, rtol=1e-7)
    assert_allclose(region.center.dec.deg, expected_centroid.dec.deg, rtol=1e-7)


def test_polygon_points_sky_region_to_pixel_to_sky():
    vertices = SkyCoord([(0, 0), (0, 1), (1, 1), (1, 0)], unit="deg")
    region = PolygonPointsSkyRegion(vertices)
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [0, 0]
    wcs.wcs.cdelt = [1, 1]
    wcs.wcs.crval = [0, 0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    pixel_region = region.to_pixel(wcs)
    assert isinstance(pixel_region, PolygonPointsPixelRegion)

    region_new = pixel_region.to_sky(wcs)

    assert isinstance(region_new, PolygonPointsSkyRegion)
    assert_allclose(region_new.vertices.ra[0], 0 * u.deg, atol=1e-7)
    assert_allclose(region_new.vertices.dec[0], 0 * u.deg, atol=1e-7)


def test_polygon_points_pixel_region_init():
    vertices = PixCoord([0, 0], [1, 1])
    region = PolygonPointsPixelRegion(vertices)
    assert region.vertices.x[0] == 0
    assert region.vertices.y[0] == 1
    assert isinstance(region.meta, RegionMeta)
    assert isinstance(region.visual, RegionVisual)


def test_polygon_points_pixel_region_rotate():
    vertices = PixCoord([0, 1], [1, 0])
    center = PixCoord(0.5, 0.5)
    region = PolygonPointsPixelRegion(vertices, center=center)
    angle = Angle(90, unit="deg")
    rotated_region = region.rotate(center, angle)
    expected_vertices = PixCoord([0, 1], [0, 1])
    assert_allclose(rotated_region.vertices.x, expected_vertices.x, atol=1e-7)
    assert_allclose(rotated_region.vertices.y, expected_vertices.y, atol=1e-7)


@requires_data()
def test_star_exclusion_no_region_bright():
    max_star_mag = -0.5
    geom = WcsGeom.create(
        skydir=(0, 0),
        binsz=0.02,
        width=(4, 4),
        frame="icrs",
        proj="CAR",
    )
    regions = extract_bright_star_regions(geom, max_star_mag=max_star_mag)
    assert len(regions) == 0


@requires_data()
def test_star_exclusion_no_region_no_stars():
    geom = WcsGeom.create(
        skydir=(253.4666667, 39.7602778),
        binsz=0.02,
        width=(4, 4),
        frame="icrs",
        proj="CAR",
    )
    regions = extract_bright_star_regions(geom)
    assert len(regions) == 0


@requires_data()
def test_star_exclusion_known_field():
    geom = WcsGeom.create(
        skydir=(83.6333, 22.0145),
        binsz=0.02,
        width=(4, 4),
        frame="icrs",
        proj="CAR",
    )
    regions = extract_bright_star_regions(geom)
    assert len(regions) == 2


def test_make_grid_rectangle_sky_regions():
    center = SkyCoord(0, 0, unit="deg", frame="galactic")
    geom = WcsGeom.create(skydir=center, frame="galactic")
    regions = make_grid_rectangle_sky_regions(
        center=center,
        width=20 * u.deg,
        height=12 * u.deg,
        wcs=geom.wcs,
        nbinx=5,
        nbiny=6,
        angle=30 * u.deg,
    )
    assert len(regions) == 30
    assert_allclose(regions[0].center.l, 350.547 * u.deg, atol=1e-2)
    assert_allclose(regions[0].center.b, -0.343 * u.deg, atol=1e-2)
    assert_allclose(regions[0].width, 4.0 * u.deg, rtol=1e-3)
    assert_allclose(regions[0].height, 2.0 * u.deg, rtol=1e-3)


def make_test_wcs():
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [100, 100]
    wcs.wcs.cdelt = [-0.1, 0.1]
    wcs.wcs.crval = [0, 0]
    wcs.wcs.ctype = ["RA---CAR", "DEC--CAR"]
    return wcs


def test_circle_sky_region_array_to_pixel():
    center = SkyCoord(0, 0, unit="deg")
    radius = [0.1, 0.2] * u.deg

    region = CircleSkyRegionArray(center=center, radius=radius)

    pixel_region = region.to_pixel(make_test_wcs())

    assert isinstance(pixel_region, CirclePixelRegionArray)

    # 0.1 deg / 0.1 deg/pix = 1 pix
    assert_allclose(pixel_region.radius, [1.0, 2.0])


def test_circle_pixel_region_array():
    # bounding box
    region = CirclePixelRegionArray(
        center=PixCoord(x=10, y=20),
        radius=np.array([1, 3, 2]),
    )

    bbox = region.bounding_box

    expected = RegionBoundingBox.from_float(
        xmin=7,
        xmax=13,
        ymin=17,
        ymax=23,
    )

    assert bbox == expected

    # area
    radius = np.array([1, 2, 3])

    region = CirclePixelRegionArray(
        center=PixCoord(x=0, y=0),
        radius=radius,
    )

    area = region.area

    expected = np.pi * radius**2

    assert area.shape == (3, 1, 1)
    assert_allclose(area[:, 0, 0], expected)

    # single_radius
    region = CirclePixelRegionArray(
        center=PixCoord(x=0, y=0),
        radius=np.array([2]),
    )

    area = region.area

    assert area.shape == (1, 1, 1)
    assert_allclose(area[0, 0, 0], np.pi * 4)


@requires_data()
def test_array_valued_regions():
    radius = [1, 2, 3, 4] * u.deg

    radius = radius.reshape((-1, 1, 1))

    circle = CircleSkyRegionArray(
        center=SkyCoord(83.63 * u.deg, 22.01 * u.deg),
        radius=radius,
    )

    axis = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=4)

    region_geom = RegionGeom.create(
        region=circle,
        axes=[axis],
    )

    assert_allclose(region_geom.center_skydir.ra, (83.63 * u.deg))
    assert_allclose(region_geom.center_skydir.dec, (22.01 * u.deg))

    assert region_geom.center_pix == (0.0, 0.0, 1.5)
    assert_allclose(
        region_geom.solid_angle().flatten(),
        [0.00095698, 0.00382794, 0.00861285, 0.01531174] * u.sr,
        rtol=1e-5,
    )
    assert_allclose(region_geom.width, [8.2, 8.2] * u.deg)

    events = EventList.read(
        "$GAMMAPY_DATA/hess-dl3-dr1/data/hess_dl3_dr1_obs_id_023523.fits.gz"
    )

    # Reference events per energy bin
    expected = []
    for (e_min, e_max), radius in zip(axis.iter_by_edges, circle.radius.flat):
        events_selected = events.select_energy(energy_range=(e_min, e_max))
        region = CircleSkyRegion(center=circle.center, radius=radius)
        events_selected = events_selected.select_region(regions=region)
        expected.append(len(events_selected.table))

    counts = RegionNDMap.from_geom(geom=region_geom)
    counts.fill_events(events)
    assert_allclose(counts.data.flatten(), expected)
    assert counts.data.shape == (4, 1, 1)
