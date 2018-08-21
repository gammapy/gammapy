# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from ..general_random import GeneralRandom


def f(x):
    return x ** 2


def test_general_random():
    general_random = GeneralRandom(f, 0, 3)
    vals = general_random.draw(random_state=42)
    assert_allclose(vals.mean(), 2.229301, atol=1e-4)
