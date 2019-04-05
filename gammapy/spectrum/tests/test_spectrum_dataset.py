# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
import astropy.units as u
import numpy as np
from ...utils.testing import requires_data, requires_dependency
from ...utils.random import get_random_state
from ...irf import EffectiveAreaTable, EnergyDispersion
from ...utils.fitting import Fit
from ...spectrum import (
    PHACountsSpectrum,
    ONOFFSpectrumDataset
)

@pytest.fixture
def effective_area():
    etrue = np.logspace(-1,1,10)*u.TeV
    return EffectiveAreaTable.from_parametrization(etrue)

@pytest.fixture
def energy_dispersion():
    etrue = np.logspace(-1,1,10)*u.TeV
    ereco = np.logspace(-1,1,5)*u.TeV
    return EnergyDispersion.from_diagonal_response(etrue, ereco)

@pytest.fixture
def on_spectrum():
    ereco = np.logspace(-1,1,5)*u.TeV
    return PHACountsSpectrum(ereco[:-1], ereco[1:], np.ones(4), backscal=np.ones(4))

@pytest.fixture
def off_spectrum():
    ereco = np.logspace(-1,1,5)*u.TeV
    return PHACountsSpectrum(ereco[:-1], ereco[1:], np.ones(4)*10, backscal=np.ones(4)*10)

@pytest.fixture
def model():
     return PowerLaw(index=2, amplitude=1e-11 / u.TeV/u.s/u.cm**2, reference=1 * u.TeV)


def test_ogip_spectrum_dataset_init(on_spectrum, off_spectrum, effective_area, energy_dispersion):
    dataset = ONOFFSpectrumDataset(ONcounts=on_spectrum, OFFcounts=off_spectrum,
                         aeff=effective_area, edisp=energy_dispersion)


    assert dataset.alpha.shape == (4,)
    assert_allclose(dataset.alpha, 0.1)

