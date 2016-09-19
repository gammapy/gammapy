# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
from ...background import RingBgMaker, ring_r_out
from ...image import SkyImageList, SkyImage
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

    # TODO: add back test
    def _test_correlate_maps(self):
        n_on = np.ones((200, 200))
        exclusion = np.ones((200, 200))
        exclusion[100:110, 100:110] = 0

        images = SkyImageList()
        images['n_on'] = SkyImage(data=n_on)
        images['a_on'] = SkyImage(data=n_on)
        images['exclusion'] = SkyImage(data=exclusion)

        r = RingBgMaker(10, 13, 1)
        r.correlate_maps(images)


def test_ring_r_out():
    actual = ring_r_out(1, 0, 1)
    assert_allclose(actual, 1)
