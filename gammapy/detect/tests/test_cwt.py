# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import pytest
from ..cwt import CWT

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@pytest.mark.skipif('not HAS_SCIPY')
def test_CWT():
    cwt = CWT(nscales=6, min_scale=6.0, scale_step=1.3)

    # TODO: run on test data
    assert 42 == 42
