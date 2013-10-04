# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import pytest
from numpy.testing import assert_allclose
from .. import utils

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@pytest.mark.skipif('not HAS_SCIPY')
def test_s_to_p():
    p = utils.s_to_p(5)
    assert_allclose(p, 2.8665157187919328e-07)
