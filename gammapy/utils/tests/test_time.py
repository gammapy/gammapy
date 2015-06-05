# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from ..time import time_ref_from_dict

def test_time_ref_from_dict():
    mjd_int = 500
    mjd_frac = 0.5
    time_ref_dict = dict(MJDREFI=mjd_int, MJDREFF=mjd_frac)
    time_ref = time_ref_from_dict(time_ref_dict)
    decimal = 4
    s_error = "time reference not compatible with defined values"
    np.testing.assert_almost_equal(time_ref.mjd, mjd_int + mjd_frac, decimal, s_error)
