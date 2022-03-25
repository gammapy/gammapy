# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from gammapy.utils.units import standardise_unit, energy_unit_format, energy_str_formatting
import astropy.units as u
import numpy as np
from gammapy.maps import MapAxis

def test_standardise_unit():
    assert standardise_unit("ph cm-2 s-1") == "cm-2 s-1"
    assert standardise_unit("ct cm-2 s-1") == "cm-2 s-1"
    assert standardise_unit("cm-2 s-1") == "cm-2 s-1"

values = [
    (1530e4 *u.keV, "15.3 GeV"),
    (1530 *u.GeV, "1.53 TeV"),
    (1530 *u.GeV, "1.53 TeV"),
    (1530.5e8 *u.keV, "153 TeV") 
]

@pytest.mark.parametrize("q, expect", values)
def test_energy_str_formatting(q, expect):
    assert energy_str_formatting(q) == expect

axis = MapAxis.from_nodes([1e-1, 200, 3.5e3, 4.6e4], name="energy", unit="GeV")
values = [
    (1.556e2 *u.TeV, "156 TeV"),
    (np.array([1e3,3.5e6,400.4e12,1512.5e12])*u.eV, ['1.00 keV', '3.50 MeV', '400 TeV','1.51 PeV']),
    ([1.54e2*u.GeV, 4300*u.keV, 300.6e12*u.eV], ['154 GeV', '4.30 MeV', '301 TeV']),
    (axis.center, ['100 MeV', '200 GeV', '3.50 TeV', '46.0 TeV']),
    ([u.Quantity(x) for x in axis.as_plot_labels ] ,['100 MeV', '200 GeV', '3.50 TeV', '46.0 TeV'])
]

@pytest.mark.parametrize("q, expect", values)
def test_energy_unit_format(q, expect):
    assert energy_unit_format(q) == expect