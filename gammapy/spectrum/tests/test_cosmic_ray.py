# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.units import Quantity
from ...utils.testing import assert_quantity_allclose
from ...spectrum import cosmic_ray_flux


def test_cosmic_ray_flux():
    energy = Quantity(1, "TeV")
    actual = cosmic_ray_flux(energy, "proton")
    desired = Quantity(0.096, "m-2 s-1 sr-1 TeV-1")
    assert_quantity_allclose(actual, desired)

    # TODO: test array quantities and other particles
