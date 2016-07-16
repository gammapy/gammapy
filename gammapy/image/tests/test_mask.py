# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
from .. import ExclusionMask
from ...utils.testing import requires_dependency


@requires_dependency('scipy')
def test_random_creation():
    exclusion = ExclusionMask.empty(nxpix=300, nypix=100)
    exclusion.fill_random_circles(n=6, max_rad=10)
    assert exclusion.mask.shape[0] == 100

    excluded = np.where(exclusion.mask == 0)
    assert excluded[0].size != 0


def test_distance_image():
    mask = ExclusionMask.empty(nxpix=300, nypix=200)
    distance = mask.distance_image
    assert distance.shape == (200, 300)
    assert_allclose(distance[0, 0],  -1.)
    assert_allclose(distance[-1, 0], -200.)
    assert_allclose(distance[0, -1], -299.001672236)
    assert_allclose(distance[-1, -1], -359.723504931)
