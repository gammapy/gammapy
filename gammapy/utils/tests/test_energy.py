# Licensed under a 3-clause BSD style license - see LICENSE.rst
from numpy.testing import assert_allclose
from gammapy.utils.energy import energy_logspace


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
