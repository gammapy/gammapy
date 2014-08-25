# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from astropy.units import Quantity
from ...spectrum import cosmic_ray_flux
from ...utils.testing import assert_quantity


def test_cosmic_ray_flux():
    energy = Quantity(1, 'TeV')
    actual = cosmic_ray_flux(energy, 'proton')
    desired = Quantity(0.096, '1 / (m2 s sr TeV)')
    assert_quantity(actual, desired)

    # TODO: test array quantities and other particles
