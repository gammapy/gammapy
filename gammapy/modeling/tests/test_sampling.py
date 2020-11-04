# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from gammapy.datasets import Datasets
from gammapy.modeling.models import Models
from gammapy.modeling.sampling import ln_uniform_prior, run_mcmc
from gammapy.utils.testing import requires_data, requires_dependency


@pytest.fixture(scope="session")
def dataset():
    filename_models = "$GAMMAPY_DATA/fermi-3fhl-crab/Fermi-LAT-3FHL_models.yaml"
    models = Models.read(filename_models)

    # Define the free parameters and min, max values
    parameters = models.parameters
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

    filename = "$GAMMAPY_DATA/fermi-3fhl-crab/Fermi-LAT-3FHL_datasets.yaml"
    datasets = Datasets.read(filename=filename)
    datasets.models = models
    return datasets


@requires_data()
def test_lnprob(dataset):
    # Testing priors and parameter bounds
    parameters = dataset.models.parameters

    # paramater is within min, max boundaries
    assert ln_uniform_prior(dataset) == 0.0
    # Setting amplitude outside min, max values
    parameters["amplitude"].value = 1000
    assert ln_uniform_prior(dataset) == -np.inf


@requires_dependency("emcee")
@requires_data()
def test_runmcmc(dataset):
    # Running a small MCMC on pregenerated datasets
    import emcee

    sampler = run_mcmc(dataset, nwalkers=6, nrun=10)  # to speedup the test
    assert isinstance(sampler, emcee.ensemble.EnsembleSampler)
