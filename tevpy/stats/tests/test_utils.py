import pytest
from numpy.testing import assert_almost_equal
from .. import utils

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

@pytest.mark.skipif('not HAS_SCIPY')
def test_s_to_p():
    p = utils.s_to_p(5)
    assert_almost_equal(p, 2.8665157187919328e-07)
