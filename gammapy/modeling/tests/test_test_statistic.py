# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
from gammapy.utils.testing import requires_data
from gammapy.modeling.test_statistic import TestStatisticNested


@pytest.fixture(scope="session")
def fermi_datasets():
    from gammapy.datasets import Datasets

    filename = "$GAMMAPY_DATA/fermi-3fhl-crab/Fermi-LAT-3FHL_datasets.yaml"
    filename_models = "$GAMMAPY_DATA/fermi-3fhl-crab/Fermi-LAT-3FHL_models.yaml"
    return Datasets.read(filename=filename, filename_models=filename_models)


@requires_data()
def test_test_statistic(fermi_datasets):

    model = fermi_datasets.models["Crab Nebula"]
    ts_eval = TestStatisticNested([model.spectral_model.amplitude], [0])
    ts = ts_eval.run(fermi_datasets)

    assert_allclose(ts, 20905.667798, rtol=1e-5)
