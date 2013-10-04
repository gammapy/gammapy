# Licensed under a 3-clause BSD style license - see LICENSE.rst
from numpy.testing import assert_almost_equal
import pytest
from ..diffuse import GalacticDiffuse

def _test_GalacticDiffuse():
    # TODO: need to download example file for test
    actual = GalacticDiffuse()(100, 30, 50)
    assert_almost_equal(actual, 0)
