# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from astropy.time import Time, TimeDelta
from ..time import time_ref_from_dict, time_relative_to_ref


def test_time_ref_from_dict():
    mjd_int = 500
    mjd_frac = 0.5
    time_ref_dict = dict(MJDREFI=mjd_int, MJDREFF=mjd_frac)
    time_ref = time_ref_from_dict(time_ref_dict)
    decimal = 4
    s_error = "time reference not compatible with defined values"
    np.testing.assert_almost_equal(time_ref.mjd, mjd_int + mjd_frac, decimal, s_error)


def test_time_relative_to_ref():
    mjd_int = 500
    mjd_frac = 0.5
    time_ref_dict = dict(MJDREFI=mjd_int, MJDREFF=mjd_frac)
    time_ref = time_ref_from_dict(time_ref_dict)
    delta_time_1sec = TimeDelta(1., format='sec')
    time = time_ref + delta_time_1sec
    delta_time = time_relative_to_ref(time, time_ref_dict)
    decimal = 4
    s_error = "delta time not compatible with defined value"
    np.testing.assert_almost_equal(delta_time.sec, delta_time_1sec.sec, decimal, s_error)
