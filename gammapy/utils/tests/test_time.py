# Licensed under a 3-clause BSD style license - see LICENSE.rst
import astropy.units as u
from numpy.testing import assert_allclose
from astropy.time import Time, TimeDelta
from gammapy.utils.time import (
    absolute_time,
    reference_time_from_header,
    reference_time_to_header,
    time_relative_to_ref,
)


def test_reference_time_from_header():
    d = dict(MJDREFI=51910, MJDREFF=0.00074287036841269583, TIMESYS="UTC")
    mjd = d["MJDREFI"] + d["MJDREFF"]

    time = reference_time_from_header(d)
    assert time.format == "mjd"
    assert time.scale == "utc"
    assert_allclose(time.mjd, mjd)


def test_reference_time_from_header_mjdref():
    mjd = 51910.0
    d = dict(MJDREF=mjd, TIMESYS="TT")

    time = reference_time_from_header(d)
    assert time.format == "mjd"
    assert time.scale == "tt"
    assert_allclose(time.mjd, mjd)


def test_reference_time_from_header_jdref():
    timeref = Time("2020-01-01T00:00:00", scale='tt')
    d = dict(JDREF=timeref.jd, TIMESYS="TT")

    time = reference_time_from_header(d)
    assert time.format == "jd"
    assert time.scale == "tt"
    assert_allclose(time.jd, timeref.jd)


def test_reference_time_to_header():
    time = Time("2001-01-01T00:00:00")

    header = reference_time_to_header(time)

    assert set(header) == {"MJDREFI", "MJDREFF", "TIMESYS"}
    assert header["MJDREFI"] == 51910
    assert_allclose(header["MJDREFF"], 0.00074287036841269583)
    assert header["TIMESYS"] == "TT"


def test_time_relative_to_ref_delta():
    time_ref = Time(500, 0.5, format='mjd')
    delta_time_1sec = TimeDelta(1.0, format="sec")
    time = time_ref + delta_time_1sec

    delta_time = time_relative_to_ref(time, time_ref)

    assert u.isclose(delta_time, delta_time_1sec.to(u.s))


def test_time_relative_to_ref_quantity():
    time_ref = Time(500, 0.5, format='mjd')
    delta_time_1sec = 1.0 * u.s

    time = absolute_time(delta_time_1sec, time_ref)
    delta_time = time_relative_to_ref(time, time_ref)

    assert u.isclose(delta_time, delta_time_1sec.to(u.s))


def test_absolute_time_quantity():
    time_ref = Time(51000, 0.5, format='mjd')
    delta_time_1sec = 1.0 * u.s
    time = time_ref + delta_time_1sec

    abs_time = absolute_time(delta_time_1sec, time_ref)
    assert time == abs_time


def test_absolute_time_delta():
    time_ref = Time(51000, 0.5, format='mjd')
    delta_time_1sec = TimeDelta(1.0 * u.s)
    time = time_ref + delta_time_1sec

    abs_time = absolute_time(delta_time_1sec, time_ref)
    assert time == abs_time
