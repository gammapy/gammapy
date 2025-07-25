# Licensed under a 3-clause BSD style license - see LICENSE.rst
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import (
    SkyCoord,
    EarthLocation,
    AltAz,
    ICRS,
    UnitSphericalRepresentation,
)
from astropy.time import Time
from gammapy.utils.coordinates import (
    FoVAltAzFrame,
    FoVICRSFrame,
    fov_to_sky,
    sky_to_fov,
)


class TestFoVAltAzFrame:
    @classmethod
    def setup_class(cls):
        # Basic setup
        location = EarthLocation(lat=45 * u.deg, lon=10 * u.deg)
        obstime = Time("2025-01-01T00:00:00")
        origin = AltAz(alt=45 * u.deg, az=120 * u.deg)

        cls.fov_frame = FoVAltAzFrame(origin=origin, obstime=obstime, location=location)
        cls.obstime = obstime
        cls.origin = origin
        cls.location = location

    def test_creation(self):
        # Construct a coordinate in FoVFrame
        fov_coord = SkyCoord(
            fov_lon=10 * u.deg, fov_lat=5 * u.deg, frame=self.fov_frame
        )

        # Check that the frame exists and properties are as expected
        assert isinstance(fov_coord.frame, FoVAltAzFrame)
        assert fov_coord.frame.obstime == self.obstime
        assert fov_coord.frame.location == self.location

        # Check representation type
        assert isinstance(fov_coord.data, UnitSphericalRepresentation)
        assert fov_coord.fov_lon.unit == u.deg
        assert fov_coord.fov_lat.unit == u.deg

    def test_altaz_transform(self):
        altaz_frame = AltAz(obstime=self.obstime + 15 * u.min, location=self.location)
        target = SkyCoord(az=[50, 60] * u.deg, alt=[20, 62] * u.deg, frame=altaz_frame)

        # Transform to FoVFrame and back
        fov = target.transform_to(self.fov_frame)
        roundtrip = fov.transform_to(altaz_frame)

        assert_allclose(roundtrip.az.deg, [50, 60])
        assert_allclose(roundtrip.alt.deg, [20, 62])

    def test_fovframe_transform(self):
        fov_frame = FoVAltAzFrame(
            origin=AltAz(alt=50 * u.deg, az=110 * u.deg),
            obstime=self.obstime + 15 * u.min,
            location=self.location,
        )
        target = SkyCoord(
            fov_lon=[50, 60] * u.deg, fov_lat=[20, 62] * u.deg, frame=fov_frame
        )

        # Transform to FoVAltAzFrame and back
        fov = target.transform_to(self.fov_frame)
        roundtrip = fov.transform_to(fov_frame)

        assert_allclose(roundtrip.fov_lon.deg, [50, 60])
        assert_allclose(roundtrip.fov_lat.deg, [20, 62])

    def test_icrs_transform(self):
        target = SkyCoord(
            ra=[10, 30, 45] * u.deg, dec=[-60, -30, 60] * u.deg, frame="icrs"
        )

        # Transform to FoVFrame and back
        fov = target.transform_to(self.fov_frame)
        roundtrip = fov.transform_to("icrs")

        assert_allclose(roundtrip.ra.deg, [10, 30, 45])
        assert_allclose(roundtrip.dec.deg, [-60, -30, 60])


def test_checked_hess_values():
    # these are cross-checked with the
    # transformation as implemented in H.E.S.S.
    fov_altaz_lon = [0.7145614, 0.86603433, -0.05409698, 2.10295248] * u.deg
    fov_altaz_lat = [-1.60829115, -1.19643974, 0.45800984, 3.26844192] * u.deg

    az_pointing = [52.42056255, 52.24706061, 52.06655505, 51.86795724] * u.deg
    alt_pointing = [51.11908203, 51.23454751, 51.35376141, 51.48385814] * u.deg
    altaz_pnt = AltAz(az=az_pointing, alt=alt_pointing)

    fov_frame = FoVAltAzFrame(origin=altaz_pnt)
    fov_coord = SkyCoord(fov_lon=fov_altaz_lon, fov_lat=fov_altaz_lat, frame=fov_frame)
    altaz = fov_coord.transform_to(altaz_pnt)
    assert_allclose(altaz.az.value, [51.320575, 50.899125, 52.154053, 48.233023])
    assert_allclose(altaz.alt.value, [49.505451, 50.030165, 51.811739, 54.700102])


class TestFoVICRSFrame:
    @classmethod
    def setup_class(cls):
        # Basic setup
        origin = ICRS(ra=45 * u.deg, dec=10 * u.deg)

        cls.fov_icrs_frame = FoVICRSFrame(origin=origin)
        cls.origin = origin

    def test_creation(self):
        # Construct a coordinate in FoVFrame
        fov_coord = SkyCoord(
            fov_lon=10 * u.deg, fov_lat=5 * u.deg, frame=self.fov_icrs_frame
        )

        # Check that the frame exists and properties are as expected
        assert isinstance(fov_coord.frame, FoVICRSFrame)

        # Check representation type
        assert isinstance(fov_coord.data, UnitSphericalRepresentation)
        assert fov_coord.fov_lon.unit == u.deg
        assert fov_coord.fov_lat.unit == u.deg

    def test_icrs_transform(self):
        target = SkyCoord(ra=[50, 60] * u.deg, dec=[20, 62] * u.deg, frame="icrs")

        # Transform to FoVICRSFrame and back
        fov_icrs = target.transform_to(self.fov_icrs_frame)
        roundtrip = fov_icrs.transform_to("icrs")

        assert_allclose(roundtrip.ra.deg, [50, 60])
        assert_allclose(roundtrip.dec.deg, [20, 62])

    def test_fovframe_transform(self):
        fov_icrs_frame = FoVICRSFrame(
            origin=SkyCoord(ra=150 * u.deg, dec=-10 * u.deg, frame="icrs")
        )

        target = SkyCoord(
            fov_lon=[50, 60] * u.deg, fov_lat=[20, 62] * u.deg, frame=fov_icrs_frame
        )

        # Transform to FoVFrame and back
        fov = target.transform_to(self.fov_icrs_frame)
        roundtrip = fov.transform_to(fov_icrs_frame)

        assert_allclose(roundtrip.fov_lon.deg, [50, 60])
        assert_allclose(roundtrip.fov_lat.deg, [20, 62])


def test_altaz_transform():
    location = EarthLocation(lat=45 * u.deg, lon=10 * u.deg)
    obstime = Time("2025-01-01T00:00:00")

    altaz_frame = AltAz(location=location, obstime=obstime)
    target = SkyCoord(
        alt=[30, 45, 75] * u.deg, az=[-60, -30, 60] * u.deg, frame=altaz_frame
    )

    origin = target[0].icrs
    fov_icrs_frame = FoVICRSFrame(origin=origin)

    # Transform to FoVFrame and back
    fov_icrs = target.transform_to(fov_icrs_frame)
    roundtrip = fov_icrs.transform_to(altaz_frame)

    assert_allclose(fov_icrs.fov_lat.deg, [0.0, 27.74973, 44.89433], atol=1e-6)

    assert_allclose(roundtrip.alt.deg, [30, 45, 75])
    assert_allclose(roundtrip.az.deg, [300, 330, 60])


def test_simple_altaz_to_fov():
    altaz_pnt = AltAz(az=0 * u.deg, alt=0 * u.deg)
    coord = SkyCoord(
        1, 1, unit="deg", frame=FoVAltAzFrame(origin=altaz_pnt)
    ).transform_to(altaz_pnt)
    assert_allclose(coord.az.value, 359)
    assert_allclose(coord.alt.value, 1)

    altaz_pnt = AltAz(az=180 * u.deg, alt=0 * u.deg)
    coord = SkyCoord(
        -1, 1, unit="deg", frame=FoVAltAzFrame(origin=altaz_pnt)
    ).transform_to(altaz_pnt)
    assert_allclose(coord.az.value, 181)
    assert_allclose(coord.alt.value, 1)

    altaz_pnt = AltAz(az=0 * u.deg, alt=60 * u.deg)
    coord = SkyCoord(
        1, 0, unit="deg", frame=FoVAltAzFrame(origin=altaz_pnt)
    ).transform_to(altaz_pnt)
    assert_allclose(coord.az.value, 358, rtol=1e-3)
    assert_allclose(coord.alt.value, 59.985, rtol=1e-3)


def test_fov_to_sky():
    # test some simple cases
    az, alt = fov_to_sky(1 * u.deg, 1 * u.deg, 0 * u.deg, 0 * u.deg)
    assert_allclose(az.value, 359)
    assert_allclose(alt.value, 1)

    az, alt = fov_to_sky(-1 * u.deg, 1 * u.deg, 180 * u.deg, 0 * u.deg)
    assert_allclose(az.value, 181)
    assert_allclose(alt.value, 1)

    az, alt = fov_to_sky(1 * u.deg, 0 * u.deg, 0 * u.deg, 60 * u.deg)
    assert_allclose(az.value, 358, rtol=1e-3)
    assert_allclose(alt.value, 59.985, rtol=1e-3)

    # these are cross-checked with the
    # transformation as implemented in H.E.S.S.
    fov_altaz_lon = [0.7145614, 0.86603433, -0.05409698, 2.10295248]
    fov_altaz_lat = [-1.60829115, -1.19643974, 0.45800984, 3.26844192]
    az_pointing = [52.42056255, 52.24706061, 52.06655505, 51.86795724]
    alt_pointing = [51.11908203, 51.23454751, 51.35376141, 51.48385814]
    az, alt = fov_to_sky(
        fov_altaz_lon * u.deg,
        fov_altaz_lat * u.deg,
        az_pointing * u.deg,
        alt_pointing * u.deg,
    )
    assert_allclose(az.value, [51.320575, 50.899125, 52.154053, 48.233023])
    assert_allclose(alt.value, [49.505451, 50.030165, 51.811739, 54.700102])


def test_sky_to_fov():
    # test some simple cases
    lon, lat = sky_to_fov(1 * u.deg, 1 * u.deg, 0 * u.deg, 0 * u.deg)
    assert_allclose(lon.value, -1)
    assert_allclose(lat.value, 1)

    lon, lat = sky_to_fov(269 * u.deg, 0 * u.deg, 270 * u.deg, 0 * u.deg)
    assert_allclose(lon.value, 1)
    assert_allclose(lat.value, 0, atol=1e-7)

    lon, lat = sky_to_fov(1 * u.deg, 60 * u.deg, 0 * u.deg, 60 * u.deg)
    assert_allclose(lon.value, -0.5, rtol=1e-3)
    assert_allclose(lat.value, 0.003779, rtol=1e-3)

    # these are cross-checked with the
    # transformation as implemented in H.E.S.S.
    az = [51.320575, 50.899125, 52.154053, 48.233023]
    alt = [49.505451, 50.030165, 51.811739, 54.700102]
    az_pointing = [52.42056255, 52.24706061, 52.06655505, 51.86795724]
    alt_pointing = [51.11908203, 51.23454751, 51.35376141, 51.48385814]
    lon, lat = sky_to_fov(
        az * u.deg, alt * u.deg, az_pointing * u.deg, alt_pointing * u.deg
    )
    assert_allclose(
        lon.value, [0.7145614, 0.86603433, -0.05409698, 2.10295248], rtol=1e-5
    )
    assert_allclose(
        lat.value, [-1.60829115, -1.19643974, 0.45800984, 3.26844192], rtol=1e-5
    )
