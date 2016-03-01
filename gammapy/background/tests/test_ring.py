# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
from astropy.io import fits
from ...background import RingBgMaker, ring_r_out
from ...data import MapsBunch
from ...utils.testing import requires_dependency


@requires_dependency('scipy')
class TestRingBgMaker:
    def test_construction(self):
        r = RingBgMaker(0.3, 0.5)
        r.info()

    def test_correlate(self):
        image = np.zeros((10, 10))
        image[5, 5] = 1
        r = RingBgMaker(3, 6, 1)
        image = r.correlate(image)
        # TODO: add assert

    def test_correlate_maps(self):
        n_on = np.ones((200, 200))
        maps = MapsBunch()
        maps['n_on'] = n_on
        maps['a_on'] = n_on
        maps['exclusion'] = n_on[100:110, 100:110] = 0
        r = RingBgMaker(10, 13, 1)
        r.correlate_maps(maps)


def test_ring_r_out():
    actual = ring_r_out(1, 0, 1)
    assert_allclose(actual, 1)
