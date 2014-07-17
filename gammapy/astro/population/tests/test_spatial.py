# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np
from numpy.testing import assert_allclose
from ....utils.distributions import normalize, density
from ..spatial import radial_distributions


def test_call():
    # TODO: Verify numbers against Papers or Axel's thesis.
    assert_allclose(radial_distributions['P90']()(1), 0.03954258779836089)



