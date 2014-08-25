# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from astropy.units import Quantity
from ...utils.testing import assert_quantity
from ...spectrum import diffuse_gamma_ray_flux


def test_diffuse_gamma_ray_flux():
    energy = Quantity(1, 'TeV')
    actual = diffuse_gamma_ray_flux(energy)
    # TODO: this is a dummy value ... needs to be implemented
    desired = Quantity(1.0, 'm^-2 s^-1 sr^-1 TeV^-1')
    assert_quantity(actual, desired)
