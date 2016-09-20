# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from ...coordinates import minimum_separation


def test_minimum_separation():
    lon1 = [0, 1, 1]
    lat1 = [0, 0, 1]
    lon2 = [1, 1]
    lat2 = [0, 0.5]
    separation = minimum_separation(lon1, lat1, lon2, lat2)
    assert_allclose(separation, [1, 0, 0.5])
