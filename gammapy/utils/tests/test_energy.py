# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from numpy.testing import assert_equal
import astropy.units as u
from ...utils.energy import EnergyBounds


def test_EnergyBounds():
    val = u.Quantity([1, 2, 3, 4, 5], "TeV")
    actual = EnergyBounds(val, "GeV")
    desired = EnergyBounds((1, 2, 3, 4, 5), "TeV")
    assert_equal(actual, desired)

    # View casting
    energy = val.view(EnergyBounds)
    actual = type(energy).__module__
    desired = "gammapy.utils.energy"
    assert_equal(actual, desired)

    # New from template
    energy = EnergyBounds([0, 1, 2, 3, 4, 5], "keV")
    energy2 = energy[1:4]
    actual = energy2
    desired = EnergyBounds([1, 2, 3], "keV")
    assert_equal(actual, desired)

    actual = energy2.nbins
    desired = 2
    assert_equal(actual, desired)

    actual = energy2.unit
    desired = u.keV
    assert_equal(actual, desired)

    # Equal log spacing
    energy = EnergyBounds.equal_log_spacing(1 * u.TeV, 10 * u.TeV, 10)
    actual = energy.nbins
    desired = 10
    assert_equal(actual, desired)

    # Log centers
    center = energy.log_centers
    actual = type(center).__module__
    desired = "gammapy.utils.energy"
    assert_equal(actual, desired)

    # Upper/lower bounds
    actual = energy.upper_bounds
    desired = energy[1:]
    assert_equal(actual, desired)

    actual = energy.lower_bounds
    desired = energy[:-1]
    assert_equal(actual, desired)

    lower = [1, 3, 4, 5]
    upper = [3, 4, 5, 8]
    actual = EnergyBounds.from_lower_and_upper_bounds(lower, upper, "TeV")
    desired = EnergyBounds([1, 3, 4, 5, 8], "TeV")
    assert_equal(actual, desired)
