# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from numpy.testing import assert_allclose
from ..pwn import PWN


def test_PWN():
    # TODO: get verified test cases
    pwn = PWN()
    assert_allclose(pwn.L(1), 1.1565942650397806e+16)
