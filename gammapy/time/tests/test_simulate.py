# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_almost_equal
from astropy.units import Quantity
from ..simulate import random_times


def test_random_times():
    # An example without dead time.
    rate = Quantity(10, "s^-1")
    time = random_times(size=100, rate=rate, random_state=0)
    assert_almost_equal(time[0].sec, 0.07958745081631101)
    assert_almost_equal(time[-1].sec, 9.186484131475076)

    # An example with `return_diff=True`
    rate = Quantity(10, "s^-1")
    time = random_times(size=100, rate=rate, return_diff=True, random_state=0)
    assert_almost_equal(time[0].sec, 0.07958745081631101)
    assert_almost_equal(time[-1].sec, 0.00047065345706976753)

    # An example with dead time.
    rate = Quantity(10, "Hz")
    dead_time = Quantity(0.1, "second")
    time = random_times(size=100, rate=rate, dead_time=dead_time, random_state=0)
    assert np.min(time) >= Quantity(0.1, "second")
    assert_almost_equal(time[0].sec, 0.1 + 0.07958745081631101)
    assert_almost_equal(time[-1].sec, 0.1 * 100 + 9.186484131475076)
