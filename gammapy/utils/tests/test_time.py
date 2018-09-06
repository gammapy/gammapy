# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from astropy.time import TimeDelta, Time
from ..time import (
    time_ref_from_dict,
    time_ref_to_dict,
    time_relative_to_ref,
    absolute_time,
)


def test_time_ref_from_dict():
    d = dict(MJDREFI=51910, MJDREFF=0.00074287036841269583)
    mjd = d["MJDREFF"] + d["MJDREFI"]

    time = time_ref_from_dict(d)
    assert time.format == "mjd"
    assert time.scale == "tt"
    assert_allclose(time.mjd, mjd)


def test_time_ref_to_dict():
    time = Time("2001-01-01T00:00:00")

    d = time_ref_to_dict(time)

    assert set(d) == {"MJDREFI", "MJDREFF", "TIMESYS"}
    assert d["MJDREFI"] == 51910
    assert_allclose(d["MJDREFF"], 0.00074287036841269583)
    assert d["TIMESYS"] == "tt"


def test_time_relative_to_ref():
    time_ref_dict = dict(MJDREFI=500, MJDREFF=0.5)
    time_ref = time_ref_from_dict(time_ref_dict)
    delta_time_1sec = TimeDelta(1., format="sec")
    time = time_ref + delta_time_1sec

    delta_time = time_relative_to_ref(time, time_ref_dict)

    assert_allclose(delta_time.sec, delta_time_1sec.sec)


def test_absolute_time():
    time_ref_dict = dict(MJDREFI=51000, MJDREFF=0.5)
    time_ref = time_ref_from_dict(time_ref_dict)
    delta_time_1sec = TimeDelta(1., format="sec")
    time = time_ref + delta_time_1sec

    abs_time = absolute_time(delta_time_1sec, time_ref_dict)

    assert abs_time.value == time.utc.isot
