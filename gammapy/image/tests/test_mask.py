# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.tests.helper import pytest
from .. import ExclusionMask
from ...utils.testing import requires_dependency


@requires_dependency('scipy')
def test_random_creation():
    exclusion = ExclusionMask.empty(nxpix=300, nypix=100)
    exclusion.fill_random_circles(n=6, max_rad=10)
    assert exclusion.mask.shape[0] == 100

    excluded = np.where(exclusion.mask == 0)
    assert excluded[0].size != 0


@pytest.mark.xfail
def test_distance_image():
    pass
    # TODO
