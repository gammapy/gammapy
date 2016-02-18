# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from ..axis import sqrt_space


def test_sqrt_space():
    values = sqrt_space(0, 2, 5)

    assert_allclose(values, [0., 1., 1.41421356, 1.73205081, 2.])
