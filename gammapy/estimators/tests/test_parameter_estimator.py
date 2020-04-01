# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
from gammapy.datasets import Datasets, SpectrumDatasetOnOff
from gammapy.estimators import ParameterEstimator
from gammapy.modeling.models import PowerLawSpectralModel, SkyModel
from gammapy.utils.testing import requires_data

pytest.importorskip("iminuit")


@pytest.fixture
def crab_datasets_1d():
    filename = "$GAMMAPY_DATA/joint-crab/spectra/hess/pha_obs23523.fits"
    dataset = SpectrumDatasetOnOff.from_ogip_files(filename)
    datasets = Datasets([dataset])
    return datasets


@pytest.fixture
def PLmodel():
    return PowerLawSpectralModel(amplitude="3e-11 cm-2s-1TeV-1", index=2.7)


@pytest.fixture
def crab_datasets_fermi():
    return Datasets.read(
        "$GAMMAPY_DATA/fermi-3fhl-crab/Fermi-LAT-3FHL_datasets.yaml",
        "$GAMMAPY_DATA/fermi-3fhl-crab/Fermi-LAT-3FHL_models.yaml",
    )


@requires_data()
def test_parameter_estimator_1d(crab_datasets_1d, PLmodel):
    datasets = crab_datasets_1d
    for dataset in datasets:
        dataset.models = SkyModel(spectral_model=PLmodel, name="Crab")

    estimator = ParameterEstimator(n_scan_values=10)

    result = estimator.run(datasets, PLmodel.amplitude, steps="all")

    assert_allclose(result["amplitude"], 5.142843823441639e-11, rtol=1e-3)
    assert_allclose(result["amplitude_err"], 6.0075e-12, rtol=1e-3)
    assert_allclose(result["ts"], 353.2092043652601, rtol=1e-3)
    assert_allclose(result["amplitude_errp"], 6.703e-12, rtol=5e-3)
    assert_allclose(result["amplitude_errn"], 6.152e-12, rtol=5e-3)

    # Add test for scan
    assert_allclose(result["amplitude_scan"].shape, 10)


@pytest.mark.xfail
@requires_data()
def test_parameter_estimator_3d(crab_datasets_fermi):
    datasets = crab_datasets_fermi
    parameter = datasets[0].models.parameters["amplitude"]
    estimator = ParameterEstimator()

    result = estimator.run(datasets, parameter, steps=["ts", "err"])

    assert_allclose(result["amplitude"], 0.328839, rtol=1e-3)
    assert_allclose(result["amplitude_err"], 0.002801, rtol=1e-3)
    assert_allclose(result["ts"], 13005.938702, rtol=1e-3)


@pytest.mark.xfail
@requires_data()
def test_parameter_estimator_3d_no_reoptimization(crab_datasets_fermi):
    datasets = crab_datasets_fermi
    parameter = datasets[0].models.parameters["amplitude"]
    estimator = ParameterEstimator(reoptimize=False, n_scan_values=10)
    alpha_value = datasets[0].models.parameters["alpha"].value

    result = estimator.run(datasets, parameter, steps="all")

    assert not datasets[0].models.parameters["alpha"].frozen
    assert_allclose(datasets[0].models.parameters["alpha"].value, alpha_value)
    assert_allclose(result["amplitude"], 0.331505, rtol=1e-4)
    assert_allclose(result["amplitude_scan"].shape, 10)
    assert_allclose(result["amplitude_scan"][0], 0.312406, atol=1e-3)
