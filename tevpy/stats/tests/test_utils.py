from numpy.testing import assert_almost_equal
from .. import utils

def test_s_to_p():
    p = utils.s_to_p(5)
    assert_almost_equal(p, 2.8665157187919328e-07)
