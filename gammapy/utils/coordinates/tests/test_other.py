# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.units import Quantity
from ...coordinates import galactic


def test_galactic():
    x = Quantity(0, "kpc")
    y = Quantity(0, "kpc")
    z = Quantity(0, "kpc")
    reference = (Quantity(8.5, "kpc"), Quantity(0, "deg"), Quantity(0, "deg"))
    assert galactic(x, y, z) == reference
