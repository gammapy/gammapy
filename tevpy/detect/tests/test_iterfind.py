# Licensed under a 3-clause BSD style license - see LICENSE.rst
from numpy.testing import assert_almost_equal
import pytest
from .. import iterfind

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@pytest.mark.skipif('not HAS_SCIPY')
def test_IterativeSourceDetector():
    pass
