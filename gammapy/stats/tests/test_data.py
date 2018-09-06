# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
import pytest
from ...stats import Stats


@pytest.fixture
def stats():
    return Stats(n_on=10, n_off=10, a_on=1, a_off=10)


def test_stats_properties(stats):
    assert_allclose(stats.alpha, stats.a_on / stats.a_off)
    assert_allclose(stats.background, stats.a_on / stats.a_off * stats.n_off)
    assert_allclose(stats.excess, stats.n_on - stats.a_on / stats.a_off * stats.n_off)


def test_stats_str(stats):
    text = str(stats)
    assert "alpha = 0.1" in text
    assert "background = 1.0" in text
    assert "excess = 9.0" in text
