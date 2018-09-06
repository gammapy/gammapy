# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
from numpy.testing import assert_allclose
import numpy as np
from astropy import units as u
from ...utils.testing import requires_dependency
from ...maps import WcsNDMap
from ...background import RingBackgroundEstimator, AdaptiveRingBackgroundEstimator


@pytest.fixture
def images():
    fov = 2.5 * u.deg

    m_ref = WcsNDMap.create(binsz=0.05, npix=201, dtype=float)
    m_ref.data += 1.
    coords = m_ref.geom.get_coord().skycoord
    center = m_ref.geom.center_skydir
    mask = coords.separation(center) < fov

    images = dict()
    images["counts"] = m_ref.copy(data=np.zeros_like(m_ref.data) + 2.)
    images["counts"].data *= mask

    images["exposure_on"] = m_ref.copy(data=np.zeros_like(m_ref.data) + 1.)
    images["exposure_on"].data *= mask

    exclusion = m_ref.copy(data=np.zeros_like(m_ref.data) + 1.)
    exclusion.data[90:110, 90:110] = 0
    images["exclusion"] = exclusion
    return images


@requires_dependency("scipy")
def test_ring_background_estimator(images):
    ring = RingBackgroundEstimator(0.35 * u.deg, 0.3 * u.deg)

    result = ring.run(images)

    in_fov = images["exposure_on"].data > 0

    assert_allclose(result["background"].data[in_fov], 2.)
    assert_allclose(result["alpha"].data[in_fov].mean(), 0.003488538457592745)
    assert_allclose(result["exposure_off"].data[in_fov].mean(), 305.1268970794541)
    assert_allclose(result["off"].data[in_fov].mean(), 610.2537941589082)

    assert_allclose(result["off"].data[~in_fov], 0.)
    assert_allclose(result["exposure_off"].data[~in_fov], 0.)
    assert_allclose(result["alpha"].data[~in_fov], 0.)


@requires_dependency("scipy")
class TestAdaptiveRingBackgroundEstimator:
    def setup(self):
        self.images = {}
        self.images["counts"] = WcsNDMap.create(binsz=0.02, npix=101, dtype=float)
        self.images["counts"].data += 1.
        self.images["exposure_on"] = WcsNDMap.create(binsz=0.02, npix=101, dtype=float)
        self.images["exposure_on"].data += 1e10
        exclusion = WcsNDMap.create(binsz=0.02, npix=101, dtype=float)
        exclusion.data += 1
        exclusion.data[40:60, 40:60] = 0
        self.images["exclusion"] = exclusion

    def test_run_const_width(self):
        ring = AdaptiveRingBackgroundEstimator(
            r_in=0.22 * u.deg, r_out_max=0.8 * u.deg, width=0.1 * u.deg
        )
        result = ring.run(self.images)

        assert_allclose(result["background"].data[50, 50], 1)
        assert_allclose(result["alpha"].data[50, 50], 0.002638522427440632)
        assert_allclose(result["exposure_off"].data[50, 50], 379 * 1e10)
        assert_allclose(result["off"].data[50, 50], 379)

        assert_allclose(result["background"].data[0, 0], 1)
        assert_allclose(result["alpha"].data[0, 0], 0.008928571428571418)
        assert_allclose(result["exposure_off"].data[0, 0], 112 * 1e10)
        assert_allclose(result["off"].data[0, 0], 112)

    def test_run_const_r_in(self):
        ring = AdaptiveRingBackgroundEstimator(
            r_in=0.22 * u.deg,
            r_out_max=0.8 * u.deg,
            width=0.1 * u.deg,
            method="fixed_r_in",
        )
        result = ring.run(self.images)

        assert_allclose(result["background"].data[50, 50], 1)
        assert_allclose(result["alpha"].data[50, 50], 0.002638522427440632)
        assert_allclose(result["exposure_off"].data[50, 50], 379 * 1e10)
        assert_allclose(result["off"].data[50, 50], 379)

        assert_allclose(result["background"].data[0, 0], 1)
        assert_allclose(result["alpha"].data[0, 0], 0.008928571428571418)
        assert_allclose(result["exposure_off"].data[0, 0], 112 * 1e10)
        assert_allclose(result["off"].data[0, 0], 112)
