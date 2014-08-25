# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from numpy.testing import assert_allclose
from ...stats import Stats


def test_Stats():
    n_on, n_off, a_on, a_off = 1, 2, 3, 4

    stats = Stats(n_on=n_on, n_off=n_off, a_on=a_on, a_off=a_off)
    assert_allclose(stats.alpha, a_on / a_off)
    assert_allclose(stats.background, a_on / a_off * n_off)
    assert_allclose(stats.excess, n_on - a_on / a_off * n_off)


def test_make_stats():
    pass


def test_combine_stats():
    pass
