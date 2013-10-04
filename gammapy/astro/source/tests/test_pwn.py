# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from numpy.testing import assert_approx_equal
from ..pwn import PWN


def test_PWN():
    # TODO: get verified test cases
    pwn = PWN()
    assert_approx_equal(pwn.L(1), 1.1565942650397806e+16)
