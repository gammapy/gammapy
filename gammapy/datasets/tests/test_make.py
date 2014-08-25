# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from astropy.coordinates import Angle
from astropy.units import Quantity
from ...utils.testing import assert_quantity
from ...datasets import make_test_psf


def test_make_test_psf_fits_table():
    # TODO
    psf = make_test_psf()
    energy = Quantity(100, 'GeV')
    theta = Angle(0.1, 'deg')
    psf2 = psf.psf_at_energy_and_theta(energy, theta)
    fraction = psf2.containment_fraction(0.1)
    assert fraction == 0.44485638998490795
