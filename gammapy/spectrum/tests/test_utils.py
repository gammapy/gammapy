# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
from astropy.units import Quantity
from astropy.tests.helper import assert_quantity_allclose
from astropy.time import Time
from astropy.table import Table
from ...utils.testing import requires_dependency
from ...spectrum import LogEnergyAxis


@requires_dependency('scipy')
def test_LogEnergyAxis():
    from scipy.stats import gmean
    energy = Quantity([1, 10, 100], 'TeV')
    energy_axis = LogEnergyAxis(energy)

    assert_allclose(energy_axis.x, [0, 1, 2])
    assert_quantity_allclose(energy_axis.energy, energy)

    energy = Quantity(gmean([1, 10]), 'TeV')
    pix = energy_axis.world2pix(energy.to('MeV'))
    assert_allclose(pix, 0.5)

    world = energy_axis.pix2world(pix)
    assert_quantity_allclose(world, energy)
