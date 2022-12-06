# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import ICRS, AltAz, SkyCoord
from astropy.time import Time
from gammapy.data import FixedPointingInfo, PointingInfo
from gammapy.data.observers import observatory_locations
from gammapy.data.pointing import PointingMode
from gammapy.utils.fits import earth_location_to_dict
from gammapy.utils.testing import assert_time_allclose, requires_data
from gammapy.utils.time import time_ref_to_dict, time_relative_to_ref


@requires_data()
class TestFixedPointingInfo:
    @classmethod
    def setup_class(cls):
        filename = "$GAMMAPY_DATA/tests/pointing_table.fits.gz"
        cls.fpi = FixedPointingInfo.read(filename)

    def test_location(self):
        lon, lat, height = self.fpi.location.geodetic
        assert_allclose(lon.deg, 16.5002222222222)
        assert_allclose(lat.deg, -23.2717777777778)
        assert_allclose(height.value, 1834.999999999783)

    def test_time_ref(self):
        expected = Time(51910.00074287037, format="mjd", scale="tt")
        assert_time_allclose(self.fpi.time_ref, expected)

    def test_time_start(self):
        time = self.fpi.time_start
        expected = Time(53025.826414166666, format="mjd", scale="tt")
        assert_time_allclose(time, expected)

    def test_time_stop(self):
        time = self.fpi.time_stop
        expected = Time(53025.844770648146, format="mjd", scale="tt")
        assert_time_allclose(time, expected)

    def test_duration(self):
        duration = self.fpi.duration
        assert_allclose(duration.sec, 1586.0000000044238)

    def test_radec(self):
        pos = self.fpi.radec
        assert_allclose(pos.ra.deg, 83.633333333333)
        assert_allclose(pos.dec.deg, 24.51444444)
        assert pos.name == "icrs"

    def test_altaz(self):
        pos = self.fpi.altaz
        assert_allclose(pos.az.deg, 7.48272)
        assert_allclose(pos.alt.deg, 41.84191)
        assert pos.name == "altaz"


@requires_data()
class TestPointingInfo:
    @classmethod
    def setup_class(cls):
        filename = "$GAMMAPY_DATA/tests/pointing_table.fits.gz"
        cls.pointing_info = PointingInfo.read(filename)

    def test_str(self):
        ss = str(self.pointing_info)
        assert "Pointing info" in ss

    def test_location(self):
        lon, lat, height = self.pointing_info.location.geodetic
        assert_allclose(lon.deg, 16.5002222222222)
        assert_allclose(lat.deg, -23.2717777777778)
        assert_allclose(height.value, 1834.999999999783)

    def test_time_ref(self):
        expected = Time(51910.00074287037, format="mjd", scale="tt")
        assert_time_allclose(self.pointing_info.time_ref, expected)

    def test_table(self):
        assert len(self.pointing_info.table) == 100

    def test_time(self):
        time = self.pointing_info.time
        assert len(time) == 100
        expected = Time(53025.826414166666, format="mjd", scale="tt")
        assert_time_allclose(time[0], expected)

    def test_duration(self):
        duration = self.pointing_info.duration
        assert_allclose(duration.sec, 1586.0000000044238)

    def test_radec(self):
        pos = self.pointing_info.radec[0]
        assert_allclose(pos.ra.deg, 83.633333333333)
        assert_allclose(pos.dec.deg, 24.51444444)
        assert pos.name == "icrs"

    def test_altaz(self):
        pos = self.pointing_info.altaz[0]
        assert_allclose(pos.az.deg, 11.45751357)
        assert_allclose(pos.alt.deg, 41.34088901)
        assert pos.name == "altaz"

    def test_altaz_from_table(self):
        pos = self.pointing_info.altaz_from_table[0]
        assert_allclose(pos.az.deg, 11.20432353385406)
        assert_allclose(pos.alt.deg, 41.37921408774436)
        assert pos.name == "altaz"

    def test_altaz_interpolate(self):
        time = self.pointing_info.time[0]
        pos = self.pointing_info.altaz_interpolate(time)
        assert_allclose(pos.az.deg, 11.45751357)
        assert_allclose(pos.alt.deg, 41.34088901)
        assert pos.name == "altaz"


def test_altaz_without_location(caplog):
    meta = {"ALT_PNT": 20.0, "AZ_PNT": 170.0}
    pointing = FixedPointingInfo(meta)

    with caplog.at_level(logging.WARNING):
        altaz = pointing.altaz
        assert altaz.alt.deg == 20.0
        assert altaz.az.deg == 170.0

    pointing = FixedPointingInfo({})

    with caplog.at_level(logging.WARNING):
        altaz = pointing.altaz
        assert np.isnan(altaz.alt.value)
        assert np.isnan(altaz.az.value)


@requires_data()
def test_read_gadf_drift():
    """Test for reading FixedPointingInfo from GADF drift eventlist"""
    pointing = FixedPointingInfo.read(
        "$GAMMAPY_DATA/hawc/crab_events_pass4/events/EventList_Crab_fHitbin5GP.fits.gz"
    )
    assert pointing.mode is PointingMode.DRIFT
    assert pointing.fixed_icrs is None
    assert isinstance(pointing.fixed_altaz, AltAz)
    assert pointing.fixed_altaz.alt == 0 * u.deg
    assert pointing.fixed_altaz.az == 0 * u.deg


@requires_data()
def test_read_gadf_pointing():
    """Test for reading FixedPointingInfo from GADF pointing eventlist"""
    pointing = FixedPointingInfo.read(
        "$GAMMAPY_DATA/magic/rad_max/data/20131004_05029747_DL3_CrabNebula-W0.40+035.fits"
    )
    assert pointing.mode is PointingMode.POINTING
    assert pointing.fixed_altaz is None
    assert isinstance(pointing.fixed_icrs.frame, ICRS)
    assert u.isclose(pointing.fixed_icrs.ra, 83.98333 * u.deg)
    assert u.isclose(pointing.fixed_icrs.dec, 22.24389 * u.deg)


def test_fixed_pointing_info_fixed_icrs():
    location = observatory_locations["cta_north"]
    pointing_icrs = SkyCoord(ra=83.28 * u.deg, dec=21.78 * u.deg)
    pointing = FixedPointingInfo(
        mode=PointingMode.POINTING,
        fixed_icrs=pointing_icrs,
    )

    # not given, but assumed if missing
    assert pointing.mode == PointingMode.POINTING
    assert pointing.fixed_icrs == pointing_icrs
    assert pointing.fixed_altaz is None

    obstime = Time("2020-10-10T03:00:00")
    altaz = pointing.get_altaz(obstime=obstime, location=location)
    icrs = pointing.get_icrs(obstime=obstime, location=location)
    back_trafo = altaz.transform_to("icrs")

    assert isinstance(altaz.frame, AltAz)
    assert altaz.obstime == obstime
    assert altaz.location == location
    assert u.isclose(back_trafo.ra, pointing_icrs.ra)
    assert u.isclose(back_trafo.dec, pointing_icrs.dec)

    assert isinstance(icrs.frame, ICRS)
    assert icrs.obstime == obstime
    assert icrs.location == location
    assert u.isclose(pointing_icrs.ra, icrs.ra)
    assert u.isclose(pointing_icrs.dec, icrs.dec)

    # test multiple times at once
    obstimes = obstime + np.linspace(0, 0.25, 50) * u.hour
    altaz = pointing.get_altaz(obstime=obstimes, location=location)
    icrs = pointing.get_icrs(obstime=obstimes, location=location)
    assert isinstance(altaz.frame, AltAz)
    assert len(altaz) == len(obstimes)
    assert np.all(altaz.obstime == obstimes)

    back_trafo = altaz.transform_to("icrs")
    assert u.isclose(back_trafo.ra, pointing_icrs.ra).all()
    assert u.isclose(back_trafo.dec, pointing_icrs.dec).all()
    assert np.all(u.isclose(pointing_icrs.ra, icrs.ra))


def test_fixed_pointing_info_fixed_altaz():
    location = observatory_locations["cta_south"]
    pointing_altaz = SkyCoord(alt=70 * u.deg, az=0 * u.deg, frame=AltAz())
    pointing = FixedPointingInfo(
        mode=PointingMode.DRIFT,
        fixed_altaz=pointing_altaz,
    )

    # not given, but assumed if missing
    assert pointing.mode == PointingMode.DRIFT
    assert pointing.fixed_icrs is None
    assert pointing.fixed_altaz == pointing_altaz

    obstime = Time("2020-10-10T03:00:00")
    altaz = pointing.get_altaz(obstime=obstime, location=location)
    icrs = pointing.get_icrs(obstime=obstime, location=location)
    back_trafo = icrs.transform_to(AltAz(location=icrs.location, obstime=icrs.obstime))

    assert isinstance(altaz.frame, AltAz)
    assert altaz.obstime == obstime
    assert altaz.location == location
    assert u.isclose(pointing_altaz.alt, altaz.alt)
    assert u.isclose(pointing_altaz.az, altaz.az, atol=1e-10 * u.deg)

    assert isinstance(icrs.frame, ICRS)
    assert icrs.obstime == obstime
    assert icrs.location == location
    assert u.isclose(back_trafo.alt, pointing_altaz.alt)
    assert u.isclose(back_trafo.az, pointing_altaz.az, atol=1e-10 * u.deg)

    # test multiple times at once
    obstimes = obstime + np.linspace(0, 0.25, 50) * u.hour
    altaz = pointing.get_altaz(obstime=obstimes, location=location)
    icrs = pointing.get_icrs(obstime=obstimes, location=location)
    assert isinstance(altaz.frame, AltAz)
    assert len(altaz) == len(obstimes)
    assert np.all(altaz.obstime == obstimes)


@pytest.mark.parametrize(
    ("obs_mode"),
    [
        ("POINTING"),
        ("WOBBLE"),
        ("SCAN"),
    ],
)
def test_fixed_pointing_info_fixed_icrs_from_meta(obs_mode):
    location = observatory_locations["cta_south"]
    start = Time("2020-11-01T03:00:00")
    stop = Time("2020-11-01T03:15:00")
    ref = Time("2020-11-01T00:00:00")
    pointing_icrs = SkyCoord(ra=83.28 * u.deg, dec=21.78 * u.deg)

    meta = time_ref_to_dict(ref)
    meta["TSTART"] = time_relative_to_ref(start, meta).to_value(u.s)
    meta["TSTOP"] = time_relative_to_ref(stop, meta).to_value(u.s)
    meta.update(earth_location_to_dict(location))
    meta["OBS_MODE"] = obs_mode
    meta["RA_PNT"] = pointing_icrs.ra.deg
    meta["DEC_PNT"] = pointing_icrs.dec.deg

    pointing = FixedPointingInfo(meta=meta)

    # not given, but assumed if missing
    assert pointing.mode == PointingMode.POINTING
    assert pointing.fixed_icrs == pointing_icrs
    assert pointing.fixed_altaz is None

    altaz = pointing.get_altaz(start)
    assert altaz.obstime == start
    assert isinstance(altaz.frame, AltAz)
    assert np.all(u.isclose(pointing_icrs.ra, pointing.get_icrs(start).ra))

    back_trafo = altaz.transform_to("icrs")
    assert u.isclose(back_trafo.ra, pointing_icrs.ra)
    assert u.isclose(back_trafo.dec, pointing_icrs.dec)

    times = start + np.linspace(0, 1, 50) * (stop - start)
    altaz = pointing.get_altaz(times)
    assert len(altaz) == len(times)
    assert np.all(altaz.obstime == times)
    assert isinstance(altaz.frame, AltAz)

    back_trafo = altaz.transform_to("icrs")
    assert u.isclose(back_trafo.ra, pointing_icrs.ra).all()
    assert u.isclose(back_trafo.dec, pointing_icrs.dec).all()
    assert np.all(u.isclose(pointing_icrs.ra, pointing.get_icrs(times).ra))


def test_fixed_pointing_info_fixed_altaz_from_meta():
    location = observatory_locations["cta_south"]
    start = Time("2020-11-01T03:00:00")
    stop = Time("2020-11-01T03:15:00")
    ref = Time("2020-11-01T00:00:00")
    pointing_icrs = SkyCoord(ra=83.28 * u.deg, dec=21.78 * u.deg)
    pointing_altaz = pointing_icrs.transform_to(AltAz(obstime=start, location=location))

    meta = time_ref_to_dict(ref)
    meta["TSTART"] = time_relative_to_ref(start, meta).to_value(u.s)
    meta["TSTOP"] = time_relative_to_ref(stop, meta).to_value(u.s)
    meta.update(earth_location_to_dict(location))
    meta["OBS_MODE"] = "DRIFT"
    meta["ALT_PNT"] = pointing_altaz.alt.deg
    meta["AZ_PNT"] = pointing_altaz.az.deg

    pointing = FixedPointingInfo(meta=meta)

    # not given, but assumed if missing
    assert pointing.mode == PointingMode.DRIFT
    assert pointing.fixed_icrs is None
    assert u.isclose(pointing.fixed_altaz.alt, pointing_altaz.alt)
    assert u.isclose(pointing.fixed_altaz.az, pointing_altaz.az)

    icrs = pointing.get_icrs(start)
    assert icrs.obstime == start
    assert isinstance(icrs.frame, ICRS)

    back_trafo = icrs.transform_to(pointing_altaz.frame)
    assert u.isclose(back_trafo.alt, pointing_altaz.alt)
    assert u.isclose(back_trafo.az, pointing_altaz.az)

    times = start + np.linspace(0, 1, 50) * (stop - start)
    icrs = pointing.get_icrs(times)
    assert len(icrs) == len(times)
    assert np.all(icrs.obstime == times)
    assert isinstance(icrs.frame, ICRS)

    back_trafo = icrs.transform_to(AltAz(location=location, obstime=times))
    assert u.isclose(back_trafo.alt, pointing_altaz.alt).all()
    assert u.isclose(back_trafo.az, pointing_altaz.az).all()

    assert np.all(u.isclose(pointing_altaz.alt, pointing.get_altaz(times).alt))
    assert np.all(u.isclose(pointing_altaz.az, pointing.get_altaz(times).az))


def test_fixed_pointing_no_meta():
    location = observatory_locations["cta_south"]
    start = Time("2020-11-01T03:00:00")
    stop = Time("2020-11-01T03:15:00")
    pointing_icrs = SkyCoord(ra=83.28 * u.deg, dec=21.78 * u.deg)

    pointing = FixedPointingInfo(
        mode=PointingMode.POINTING,
        fixed_icrs=pointing_icrs,
        location=location,
        time_start=start,
        time_stop=stop,
    )

    assert pointing.mode == PointingMode.POINTING
    assert pointing.fixed_icrs == pointing_icrs
    assert pointing.fixed_altaz is None

    altaz = pointing.get_altaz(start)
    assert altaz.obstime == start
    assert isinstance(altaz.frame, AltAz)
    assert np.all(u.isclose(pointing_icrs.ra, pointing.get_icrs(start).ra))

    back_trafo = altaz.transform_to("icrs")
    assert u.isclose(back_trafo.ra, pointing_icrs.ra)
    assert u.isclose(back_trafo.dec, pointing_icrs.dec)

    times = start + np.linspace(0, 1, 50) * (stop - start)
    altaz = pointing.get_altaz(times)
    assert len(altaz) == len(times)
    assert np.all(altaz.obstime == times)
    assert isinstance(altaz.frame, AltAz)

    back_trafo = altaz.transform_to("icrs")
    assert u.isclose(back_trafo.ra, pointing_icrs.ra).all()
    assert u.isclose(back_trafo.dec, pointing_icrs.dec).all()
    assert np.all(u.isclose(pointing_icrs.ra, pointing.get_icrs(times).ra))
