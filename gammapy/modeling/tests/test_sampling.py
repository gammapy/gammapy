# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from gammapy.datasets import Datasets
from gammapy.modeling.sampling import run_mcmc, ln_uniform_prior
import numpy as np
#@pytest.fixture
def get_datasets():
    dataset = Datasets.read("$GAMMAPY_DATA/fermi-3fhl-crab/Fermi-LAT-3FHL_datasets.yaml",
                            "$GAMMAPY_DATA/fermi-3fhl-crab/Fermi-LAT-3FHL_models.yaml")

    # Define the free parameters and min, max values
    parameters = dataset.models.parameters
    parameters["lon_0"].frozen = False
    parameters["lat_0"].frozen = False
    parameters["norm"].frozen = True
    parameters["alpha"].frozen = True
    parameters["beta"].frozen = True
    parameters["lat_0"].min = -90
    parameters["lat_0"].max = 90
    parameters["lon_0"].min = 0
    parameters["lon_0"].max = 360
    parameters["amplitude"].min = 0.01 * parameters["amplitude"].value
    parameters["amplitude"].max = 100 * parameters["amplitude"].value

    return dataset

def test_lnprob():
    #Testing priors
    dataset = get_datasets()
    parameters = dataset.models.parameters

    #paramater is within min, max boundaries
    assert ln_uniform_prior(dataset) == 0.0
    # Setting amplitude outside min, max values
    parameters["amplitude"].value = 1000
    assert ln_uniform_prior(dataset) == -np.inf

def test_runmcmc():
    #Running a small MCMC on pregenerated datasets
    dataset = get_datasets()
    sampler = run_mcmc(dataset, nwalkers=6, nrun=10)  # to speedup the test


