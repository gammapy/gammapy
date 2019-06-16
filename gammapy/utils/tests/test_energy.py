# Licensed under a 3-clause BSD style license - see LICENSE.rst
from astropy import units as u
from numpy.testing import assert_allclose
from ..energy import energy_logcenter, energy_logspace


def test_energy_logspace():
    energy = energy_logspace(emin="0.1 TeV", emax="10 TeV", nbins=3)
    assert energy.unit == "TeV"
    assert_allclose(energy.value, [0.1, 1, 10])

    energy = energy_logspace(emin=0.1, emax=10, nbins=3, unit="TeV")
    assert energy.unit == "TeV"
    assert_allclose(energy.value, [0.1, 1, 10])

    energy = energy_logspace(emin="0.1 TeV", emax="10 TeV", nbins=1, per_decade=True)
    assert energy.unit == "TeV"
    assert_allclose(energy.value, [0.1, 10])


def test_energy_locenter():
    e_edges = [0.01, 1, 100] * u.TeV
    e_center = energy_logcenter(e_edges=e_edges)
    assert e_center.unit == "TeV"
    assert_allclose(e_center.value, [0.1, 10])
