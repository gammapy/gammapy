# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import unittest
from astropy.tests.helper import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.io import fits
from ...background import Maps, RingBgMaker, ring_r_out

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@pytest.mark.skipif('not HAS_SCIPY')
class TestRingBgMaker(unittest.TestCase):
    def test_construction(self):
        r = RingBgMaker(0.3, 0.5)
        r.info()

    def test_correlate(self):
        image = np.zeros((10, 10))
        image[5, 5] = 1
        r = RingBgMaker(3, 6, 1)
        image = r.correlate(image)

    def test_correlate_maps(self):
        n_on = np.ones((200, 200))
        hdu = fits.ImageHDU(n_on, name='n_on')
        maps = Maps([hdu])
        maps['exclusion'].data[100:110, 100:110] = 0
        r = RingBgMaker(10, 13, 1)
        r.correlate_maps(maps)


class TestHelperFuntions(unittest.TestCase):
    def test_compute_r_o(self):
        actual = ring_r_out(1, 0, 1)
        assert_allclose(actual, 1)
