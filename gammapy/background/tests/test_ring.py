# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from astropy import units as u
from ...background import (RingBackgroundEstimator, AdaptiveRingBackgroundEstimator,
                           ring_r_out)
from ...image import SkyImageList, SkyImage
from ...utils.testing import requires_dependency


@requires_dependency('scipy')
class TestRingBackgroundEstimator:
    def setup(self):
        self.ring = RingBackgroundEstimator(0.1 * u.deg, 0.1 * u.deg)
        self.images = SkyImageList()

        self.images['counts'] = SkyImage.empty(nxpix=101, nypix=101, fill=1)
        self.images['exposure_on'] = SkyImage.empty(nxpix=101, nypix=101, fill=1e10)
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


@requires_dependency('scipy')
class TestAdaptiveRingBackgroundEstimator:
    def setup(self):
        opts = dict(r_in=0.22 * u.deg, r_out_max=0.8 * u.deg, width=0.1 * u.deg)
        self.ring = AdaptiveRingBackgroundEstimator(**opts)
        self.images = SkyImageList()

        self.images['counts'] = SkyImage.empty(nxpix=101, nypix=101, fill=1)
        self.images['exposure_on'] = SkyImage.empty(nxpix=101, nypix=101, fill=1e10)
        exclusion = SkyImage.empty(nxpix=101, nypix=101, fill=1)
        exclusion.data[40:60, 40:60] = 0
        self.images['exclusion'] = exclusion

    def test_run_const_width(self):
        result = self.ring.run(self.images)
        assert_allclose(result['background'].data[50, 50], 1)
        assert_allclose(result['alpha'].data[50, 50], 0.002638522427440632)
        assert_allclose(result['exposure_off'].data[50, 50], 379 * 1e10)
        assert_allclose(result['off'].data[50, 50], 379)

        assert_allclose(result['background'].data[0, 0], 1)
        assert_allclose(result['alpha'].data[0, 0], 0.008928571428571418)
        assert_allclose(result['exposure_off'].data[0, 0], 112 * 1e10)
        assert_allclose(result['off'].data[0, 0], 112)

    def test_run_const_r_in(self):
        self.ring.parameters['method'] = 'fixed_r_in'
        result = self.ring.run(self.images)
        assert_allclose(result['background'].data[50, 50], 1)
        assert_allclose(result['alpha'].data[50, 50], 0.002638522427440632)
        assert_allclose(result['exposure_off'].data[50, 50], 379 * 1e10)
        assert_allclose(result['off'].data[50, 50], 379)

        assert_allclose(result['background'].data[0, 0], 1)
        assert_allclose(result['alpha'].data[0, 0], 0.008928571428571418)
        assert_allclose(result['exposure_off'].data[0, 0], 112 * 1e10)
        assert_allclose(result['off'].data[0, 0], 112)


def test_ring_r_out():
    actual = ring_r_out(1, 0, 1)
    assert_allclose(actual, 1)
