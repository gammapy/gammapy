# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_almost_equal
from astropy.time import TimeDelta
from ..time import time_ref_from_dict, time_relative_to_ref, absolute_time


def test_time_ref_from_dict():
    mjd_int = 500
    mjd_frac = 0.5
    time_ref_dict = dict(MJDREFI=mjd_int, MJDREFF=mjd_frac)

    time_ref = time_ref_from_dict(time_ref_dict)

    assert_almost_equal(time_ref.mjd, mjd_int + mjd_frac, decimal=4)


def test_time_relative_to_ref():
    time_ref_dict = dict(MJDREFI=500, MJDREFF=0.5)
    time_ref = time_ref_from_dict(time_ref_dict)
    delta_time_1sec = TimeDelta(1., format='sec')
    time = time_ref + delta_time_1sec

    delta_time = time_relative_to_ref(time, time_ref_dict)

    assert_almost_equal(delta_time.sec, delta_time_1sec.sec, decimal=4)


def test_absolute_time():
    time_ref_dict = dict(MJDREFI=500, MJDREFF=0.5)
    time_ref = time_ref_from_dict(time_ref_dict)
    delta_time_1sec = TimeDelta(1., format='sec')
    time = time_ref + delta_time_1sec

    abs_time = absolute_time(delta_time_1sec, time_ref_dict)

    assert abs_time.value == time.utc.isot
