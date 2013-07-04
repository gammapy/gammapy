import numpy as np
import unittest
from numpy.testing import assert_almost_equal
from astrometry import gal2equ, equ2gal, separation


class TestAstrometry(unittest.TestCase):

    def test_coordinate_conversion_1(self):
        """Show that it works for the Galactic center by
        comparing to the results from:
        http://heasarc.gsfc.nasa.gov/cgi-bin/Tools/convcoord/convcoord.pl"""
        assert_almost_equal(gal2equ(0, 0),
                            (266.404996, -28.936172),
                            decimal=3)

    def test_coordinate_conversion_array(self):
        """Make a grid of positions and make a closure test"""
        # Note: points near the pole or RA boundary don't work!
        x, y = np.mgrid[1:360:10, -89:90:1]
        for back, forth in [[equ2gal, gal2equ],
                            [gal2equ, equ2gal]]:
            a, b = forth(x, y)
            X, Y = back(a, b)
            assert_almost_equal((X, Y),
                                (x, y))

    def test_separation(self):
        """Test of few simple cases"""
        assert_almost_equal(separation(0, 0, 180, 0), 180)
        assert_almost_equal(separation(270, 0, 180, 0), 90)
        assert_almost_equal(separation(0, 0, 0, 90), 90)
        assert_almost_equal(separation(0, 89, 180, 89), 2)

if __name__ == "__main__":
    unittest.main()
