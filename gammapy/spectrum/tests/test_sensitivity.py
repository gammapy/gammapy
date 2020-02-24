# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from gammapy.irf import EDispKernel, EffectiveAreaTable
from gammapy.datasets import SpectrumDataset, SpectrumDatasetOnOff
from gammapy.maps import CountsSpectrum
from gammapy.spectrum import SensitivityEstimator


@pytest.fixture()
def spectrum_dataset():
    e_true = np.logspace(0, 1, 21) * u.TeV
    e_reco = np.logspace(0, 1, 5) * u.TeV
    aeff = EffectiveAreaTable.from_constant(value=1e6 * u.m ** 2, energy=e_true)
    edisp = EDispKernel.from_diagonal_response(e_true, e_reco)

    data = 3600 * np.ones(4)
    data[-1] *= 1e-3
    background = CountsSpectrum(energy_lo=e_reco[:-1], energy_hi=e_reco[1:], data=data)
    return SpectrumDataset(aeff=aeff, livetime="1h", edisp=edisp, background=background)


def test_cta_sensitivity_estimator(spectrum_dataset):
    dataset_on_off = SpectrumDatasetOnOff.from_spectrum_dataset(
        dataset=spectrum_dataset, acceptance=1, acceptance_off=5
    )
    sens = SensitivityEstimator(gamma_min=20)
    table = sens.run(dataset_on_off)

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
