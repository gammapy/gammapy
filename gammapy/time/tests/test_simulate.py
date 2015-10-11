# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_almost_equal
from astropy.units import Quantity
from ..simulate import make_random_times_poisson_process as random_times


def test_make_random_times_poisson_process():
    time = random_times(size=10,
                        rate=Quantity(10, 'Hz'),
                        dead_time=Quantity(0.1, 'second'),
                        random_state=0)
    assert np.min(time) >= Quantity(0.1, 'second')
    assert_almost_equal(time[0].sec, 0.179587450816311)
    assert_almost_equal(time[-1].sec, 0.14836021009022532)
