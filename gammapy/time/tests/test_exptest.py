# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from astropy.units import Quantity
from ..exptest import exptest
from ..simulate import random_times


def test_exptest():
    rate = Quantity(10, "s^-1")
    time_delta = random_times(100, rate=rate, return_diff=True, random_state=0)
    mr = exptest(time_delta)
    assert_allclose(mr, 0.11395763079)
