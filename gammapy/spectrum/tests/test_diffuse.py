# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from numpy.testing import assert_allclose
from ..diffuse import GalacticDiffuse

def _test_GalacticDiffuse():
    # TODO: need to download example file for test
    actual = GalacticDiffuse()(100, 30, 50)
    assert_allclose(actual, 0)
