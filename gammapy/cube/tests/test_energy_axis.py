# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from numpy.testing import assert_equal
import astropy.units as u
from ...cube import EnergyAxis

def test_EnergyAxis():
    val = u.Quantity([1, 2, 3, 4, 5], "TeV")
    actual = EnergyAxis(val, "GeV")
    desired = EnergyAxis((1, 2, 3, 4, 5), "TeV")
    assert_equal(actual, desired)

    # New from template
    energy = EnergyAxis([0, 1, 2, 3, 4, 5], "keV")
    actual = energy.nbin
    desired = 5
    assert_equal(actual, desired)

    actual = energy.unit
    desired = u.keV
    assert_equal(actual, desired)

    energy = EnergyAxis([0, 1, 2, 3, 4, 5], "keV", node_type='center')
    actual = energy.nbin
    desired = 6
    assert_equal(actual, desired)

   # Equal log spacing
    energy = EnergyAxis.equal_log_spacing(1 * u.TeV, 10 * u.TeV, 10)
    actual = energy.nbin
    desired = 10
    assert_equal(actual, desired)

 