# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
import numpy as np
from numpy.testing import assert_allclose
from ..celestial import gal2equ, equ2gal, separation


def test_coordinate_conversion_scalar():
    """Show that it works for the Galactic center by
    comparing to the results from:
    http://heasarc.gsfc.nasa.gov/cgi-bin/Tools/convcoord/convcoord.pl"""
    actual = gal2equ(0, 0)
    expected = (266.404996, -28.936172)
    assert_allclose(actual, expected, atol=3)


def test_coordinate_conversion_array():
    """Make a grid of positions and make a closure test"""
    # Note: points near the pole or RA boundary don't work!
    x, y = np.mgrid[1:360:10, -89:90:1]
    for back, forth in [[equ2gal, gal2equ],
                        [gal2equ, equ2gal]]:
        a, b = forth(x, y)
        X, Y = back(a, b)
        assert_allclose((X, Y), (x, y), atol=1e-10)


def test_separation():
    assert_allclose(separation(0, 0, 180, 0), 180)
    assert_allclose(separation(270, 0, 180, 0), 90)
    assert_allclose(separation(0, 0, 0, 90), 90)
    assert_allclose(separation(0, 89, 180, 89), 2)
