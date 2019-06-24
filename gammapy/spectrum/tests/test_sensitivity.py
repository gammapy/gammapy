# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
import numpy as np
import astropy.units as u
from ...utils.testing import requires_data
from ...irf import EffectiveAreaTable, EnergyDispersion
from ..core import CountsSpectrum
from ..sensitivity import SensitivityEstimator


@pytest.fixture()
def sens():
    etrue = np.logspace(0, 1, 21) * u.TeV
    elo = etrue[:-1]
    ehi = etrue[1:]
    area = np.zeros(20) + 1e6 * u.m ** 2

    arf = EffectiveAreaTable(energy_lo=elo, energy_hi=ehi, data=area)

    ereco = np.logspace(0, 1, 5) * u.TeV
    rmf = EnergyDispersion.from_diagonal_response(etrue, ereco)

    bkg_array = np.ones(4)
    bkg_array[-1] = 1e-3
    bkg = CountsSpectrum(
        energy_lo=ereco[:-1], energy_hi=ereco[1:], data=bkg_array, unit="s-1"
    )

    sens = SensitivityEstimator(
        arf=arf, rmf=rmf, bkg=bkg, livetime=1 * u.h, index=2, gamma_min=20, alpha=0.2
    )
    sens.run()
    return sens


@requires_data()
def test_cta_sensitivity_estimator(sens):
    table = sens.results_table

    assert len(table) == 4
    assert table.colnames == ["energy", "e2dnde", "excess", "background", "criterion"]
    assert table["energy"].unit == "TeV"
    assert table["e2dnde"].unit == "erg / (cm2 s)"

    row = table[0]
    assert_allclose(row["energy"], 1.33352, rtol=1e-3)
    assert_allclose(row["e2dnde"], 3.40101e-11, rtol=1e-3)
    assert_allclose(row["excess"], 334.454, rtol=1e-3)
    assert_allclose(row["background"], 3600, rtol=1e-3)
    assert row["criterion"] == "significance"

    row = table[3]
    assert_allclose(row["energy"], 7.49894, rtol=1e-3)
    assert_allclose(row["e2dnde"], 1.14367e-11, rtol=1e-3)
    assert_allclose(row["excess"], 20, rtol=1e-3)
    assert_allclose(row["background"], 3.6, rtol=1e-3)
    assert row["criterion"] == "gamma"
