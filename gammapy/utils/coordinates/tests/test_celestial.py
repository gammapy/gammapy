# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np
from numpy.testing import assert_allclose
from .. import celestial


def test_coordinate_conversion_scalar():
    """Show that it works for the Galactic center by
    comparing to the results from:
    http://heasarc.gsfc.nasa.gov/cgi-bin/Tools/convcoord/convcoord.pl
    """
    actual = celestial.galactic_to_radec(0, 0)
    expected = (266.404996, -28.936172)
    assert_allclose(actual, expected, atol=1e-4)


def test_coordinate_conversion_array():
    """Make a grid of positions and make a closure test"""
    # Note: points near the pole or RA boundary don't work!
    x, y = np.mgrid[1:360:10, -89:90:1]
    for back, forth in [[celestial.radec_to_galactic, celestial.galactic_to_radec],
                        [celestial.galactic_to_radec, celestial.radec_to_galactic]]:
        a, b = forth(x, y)
        X, Y = back(a, b)
        assert_allclose((X, Y), (x, y), atol=1e-10)


def test_sky_to_sky():
    actual = celestial.sky_to_sky(0, 0, 'galactic', 'icrs')
    expected = (266.404996, -28.936172)
    assert_allclose(actual, expected, atol=1e-4)
