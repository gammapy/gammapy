# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
from gammapy.datasets import Datasets, SpectrumDatasetOnOff
from gammapy.estimators.parameter import ParameterEstimator
from gammapy.modeling.models import PowerLawSpectralModel, SkyModel
from gammapy.utils.testing import requires_data

pytest.importorskip("iminuit")


@pytest.fixture
def crab_datasets_1d():
    filename = "$GAMMAPY_DATA/joint-crab/spectra/hess/pha_obs23523.fits"
    dataset = SpectrumDatasetOnOff.read(filename)
    datasets = Datasets([dataset])
    return datasets


@pytest.fixture
def PLmodel():
    return PowerLawSpectralModel(amplitude="3e-11 cm-2s-1TeV-1", index=2.7)


@pytest.fixture
def crab_datasets_fermi():
    filename = "$GAMMAPY_DATA/fermi-3fhl-crab/Fermi-LAT-3FHL_datasets.yaml"
    filename_models = "$GAMMAPY_DATA/fermi-3fhl-crab/Fermi-LAT-3FHL_models.yaml"

    return Datasets.read(filename=filename, filename_models=filename_models)


@requires_data()
def test_parameter_estimator_1d(crab_datasets_1d, PLmodel):
    datasets = crab_datasets_1d
    for dataset in datasets:
        dataset.models = SkyModel(spectral_model=PLmodel, name="Crab")

    estimator = ParameterEstimator(scan_n_values=10, selection_optional="all")

    result = estimator.run(datasets, parameter="amplitude")

    assert_allclose(result["amplitude"], 5.1428e-11, rtol=1e-3)
    assert_allclose(result["amplitude_err"], 6.42467e-12, rtol=1e-3)
    assert_allclose(result["ts"], 353.2092, rtol=1e-3)
    assert_allclose(result["amplitude_errp"], 6.703e-12, rtol=5e-3)
    assert_allclose(result["amplitude_errn"], 6.152e-12, rtol=5e-3)

    # Add test for scan
    assert_allclose(result["amplitude_scan"].shape, 10)


@requires_data()
def test_parameter_estimator_3d_no_reoptimization(crab_datasets_fermi):
    datasets = crab_datasets_fermi
    parameter = datasets[0].models.parameters["amplitude"]
    estimator = ParameterEstimator(reoptimize=False, scan_n_values=10, selection_optional=["scan"])
    alpha_value = datasets[0].models.parameters["alpha"].value

    result = estimator.run(datasets, parameter)

    assert not datasets[0].models.parameters["alpha"].frozen
    assert_allclose(datasets[0].models.parameters["alpha"].value, alpha_value)
    assert_allclose(result["amplitude"], 0.018251, rtol=1e-3)
    assert_allclose(result["amplitude_scan"].shape, 10)
    assert_allclose(result["amplitude_scan"][0], 0.017282, atol=1e-3)
