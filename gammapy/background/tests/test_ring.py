# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
from astropy import units as u
from ...background import RingBackgroundEstimator, ring_r_out
from ...image import SkyImageList, SkyImage
from ...utils.testing import requires_dependency


@requires_dependency('scipy')
class TestRingBackgroundEstimator:
    def setup(self):
        self.ring = RingBackgroundEstimator(0.1 * u.deg, 0.1 * u.deg)
        self.images = SkyImageList()

        self.images['counts'] = SkyImage.empty(nxpix=101, nypix=101, fill=1)
        self.images['exposure_on'] = SkyImage.empty(nxpix=101, nypix=101, fill=1E10)
        exclusion = SkyImage.empty(nxpix=101, nypix=101, fill=1)
        exclusion.data[40:60, 40:60] = 0
        self.images['exclusion'] = exclusion


    def test_run(self):
        result = self.ring.run(self.images)
        assert_allclose(result['background'].data[50, 50], 1)
        assert_allclose(result['alpha'].data[50, 50], 0.5)
        assert_allclose(result['exposure_off'].data[50, 50], 20000000000.0)
        assert_allclose(result['off'].data[50, 50], 2)

        assert_allclose(result['background'].data[0, 0], 1)
        assert_allclose(result['alpha'].data[0, 0], 0.004032258064516129)
        assert_allclose(result['exposure_off'].data[0, 0], 2480000000000.0)
        assert_allclose(result['off'].data[0, 0], 248.0)



def test_ring_r_out():
    actual = ring_r_out(1, 0, 1)
    assert_allclose(actual, 1)
