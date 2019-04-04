# Licensed under a 3-clause BSD style license - see LICENSE.rst
import astropy.units as u
import numpy as np
from numpy.testing import assert_allclose
from ...utils.testing import requires_data, requires_dependency
from ...utils.random import get_random_state
from ...irf import EffectiveAreaTable, EnergyDispersion
from ...utils.fitting import Fit
from ...spectrum import (
    CountsSpectrum,
    PHACountsSpectrum,
    SpectrumObservationList,
    SpectrumObservation,
    SpectrumFit,
    SpectrumFitResult,
    models,
    SpectrumDataset,
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


def test_ogip_spectrum_dataset_init(on_spectrum, off_spectrum, effective_area, energy_dispersion):
