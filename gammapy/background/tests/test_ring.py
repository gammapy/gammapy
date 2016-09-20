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
        self.ring = RingBackgroundEstimator(0.35 * u.deg, 0.3 * u.deg)
        self.images = SkyImageList.read('$GAMMAPY_EXTRA/test_datasets/unbundled/'
                                        'poisson_stats_image/input_all.fits.gz')
        self.images['exposure'].name = 'exposure_on'

    def test_run(self):
        result = self.ring.run(self.images)
        assert_allclose(result['background'].data[100, 100], 1.00822737472)
        assert_allclose(result['alpha'].data[100, 100], 0.00074794315632)
        assert_allclose(result['exposure_off'].data[100, 100], 1.33699999452e+15)
        assert_allclose(result['off'].data[100, 100], 1348)


def test_ring_r_out():
    actual = ring_r_out(1, 0, 1)
    assert_allclose(actual, 1)
