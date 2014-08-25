# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np
from numpy.testing import assert_allclose
from ...coordinates import (galactic_to_radec,
                            radec_to_galactic,
                            separation,
                            minimum_separation,
                            )


def test_coordinate_conversion_scalar():
    """Show that it works for the Galactic center by
    comparing to the results from:
    http://heasarc.gsfc.nasa.gov/cgi-bin/Tools/convcoord/convcoord.pl
    """
    actual = galactic_to_radec(0, 0)
    expected = (266.404996, -28.936172)
    assert_allclose(actual, expected, atol=1e-4)


def test_coordinate_conversion_array():
    """Make a grid of positions and make a closure test"""
    # Note: points near the pole or RA boundary don't work!
    y, x = np.mgrid[-89:90:1, 1:360:10]
    for back, forth in [[radec_to_galactic, galactic_to_radec],
                        [galactic_to_radec, radec_to_galactic]]:
        a, b = forth(x, y)
        X, Y = back(a, b)
        assert_allclose((X, Y), (x, y), atol=1e-10)


def test_separation():
    assert_allclose(separation(0, 0, 180, 0), 180)
    assert_allclose(separation(270, 0, 180, 0), 90)
    assert_allclose(separation(0, 0, 0, 90), 90)
    assert_allclose(separation(0, 89, 180, 89), 2)


def test_minimum_separation():
    lon1 = [0, 1, 1]
    lat1 = [0, 0, 1]
    lon2 = [1, 1]
    lat2 = [0, 0.5]
    separation = minimum_separation(lon1, lat1, lon2, lat2)
    assert_allclose(separation, [1, 0, 0.5])


def test_pair_correlation():
    pass
