# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
from astropy.units import Quantity
from ..exptest import exptest
from ..simulate import make_random_times_poisson_process


def test_exptest():
    rate = Quantity(100, 's^-1')
    time_delta = make_random_times_poisson_process(1000, rate=rate, random_state=0)
    mr = exptest(time_delta)
    assert_allclose(mr, 1.3790634240947202)


