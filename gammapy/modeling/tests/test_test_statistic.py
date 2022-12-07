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
def test_test_statistic_detection(fermi_datasets):

    model = fermi_datasets.models["Crab Nebula"]
    ts_eval = TestStatisticNested([model.spectral_model.amplitude], [0])
    ts = ts_eval.run(fermi_datasets)

    assert_allclose(ts, 20905.667798, rtol=1e-5)


@requires_data()
def test_test_statistic_link(fermi_datasets):

    # TODO: better test with simuated data ?
    model = fermi_datasets.models["Crab Nebula"]
    model2 = model.copy(name="other")
    model2.spectral_model.alpha.value = 2.4
    fermi_datasets.models = fermi_datasets.models + [model2]

    ts_eval = TestStatisticNested(
        [model.spectral_model.alpha], [model2.spectral_model.alpha]
    )
    ts = ts_eval.run(fermi_datasets)

    assert_allclose(ts, -3.602895, rtol=1e-5)
    assert_allclose(model2.spectral_model.alpha.value, model.spectral_model.alpha.value)
