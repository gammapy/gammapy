# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
import astropy.units as u
from ...cube import EnergyAxis

def test_EnergyAxis():
    val = u.Quantity([1, 2, 3, 4, 5], "TeV")
    actual = EnergyAxis(val, "GeV")
    desired = EnergyAxis((1, 2, 3, 4, 5), "TeV")
    assert_equal(actual, desired)

    # Test log spaced
    energy = EnergyAxis([0, 1, 2, 3, 4, 5], "keV")
    assert_equal(energy.nbin, 5)
    assert_equal(energy.unit, u.keV)

    # Test lin spaced
    energy = EnergyAxis([0, 1, 2, 3, 4, 5], "keV", interp='lin')
    assert_equal(energy.center.to_value("keV"),[0.5, 1.5, 2.5, 3.5, 4.5])

    energy = EnergyAxis([0, 1, 2, 3, 4, 5], "keV", node_type='center')
    assert_equal(energy.nbin, 6)

    # Test unit
    with pytest.raises(ValueError):
        EnergyAxis([0, 1, 2, 3, 4, 5], "m", node_type='center')


def test_from_lower_and_upper_bounds():
    lower_bounds = [1, 2, 3, 4]
    upper_bounds = [2, 3, 4, 5]
    actual = EnergyAxis.from_lower_and_upper_bounds(lower_bounds, upper_bounds, unit='TeV')
    desired = EnergyAxis([1, 2, 3, 4, 5], "TeV")
    assert_equal(actual, desired)

    actual = EnergyAxis.from_lower_and_upper_bounds(lower_bounds, upper_bounds*u.GeV)
    desired = EnergyAxis([1, 2, 3, 4, 5], "GeV")
    assert_equal(actual, desired)


def test_equal_log_spacing():
    energy = EnergyAxis.equal_log_spacing(1 * u.TeV, 10 * u.TeV, 10)
    assert_equal(energy.nbin, 10)

    energy = EnergyAxis.equal_log_spacing(1 * u.TeV, 10 * u.TeV, 10, node_type='center')
    assert_equal(energy.nbin, 10)
    assert_allclose(energy.center.to_value('TeV'), np.logspace(0.,1.,10))

def test_contains():
    # First for an edge based axis
    energy = EnergyAxis([1, 2, 3, 4, 5], "TeV")
    test_energies = u.Quantity([0.5, 3.5, 5.6], "TeV")
    assert_equal(energy.contains(test_energies), [False, True, False])

    # Now for a center based axis with linear interpolation
    energy = EnergyAxis([1, 2, 3, 4, 5], "TeV", node_type='center', interp='lin')
    assert_equal(energy.contains(test_energies), [True, True, False])

def test_find_energy_bin():
    # First for an edge based axis
    energy = EnergyAxis([1, 2, 3, 4, 5], "TeV")
    test_energies = u.Quantity([1.5, 3.5], "TeV")
    bins = energy.find_energy_bin(test_energies)
    assert_allclose(bins, [0, 2])

    test_energies = u.Quantity([0.5, 3.5, 5.6], "TeV")
    with pytest.raises(ValueError):
        bins = energy.find_energy_bin(test_energies)
