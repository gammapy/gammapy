# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
from gammapy.utils.testing import requires_data
from gammapy.datasets import SpectrumDatasetOnOff, Datasets
from gammapy.modeling.models import PowerLawSpectralModel, SkyModel
from gammapy.cube import ParameterEstimator

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
    return Datasets.read("$GAMMAPY_DATA/fermi-3fhl-crab/Fermi-LAT-3FHL_datasets.yaml",
                                   "$GAMMAPY_DATA/fermi-3fhl-crab/Fermi-LAT-3FHL_models.yaml")

@requires_data()
def test_parameter_estimator_1d(crab_datasets_1d, PLmodel):
    datasets = crab_datasets_1d
    for dataset in datasets:
        dataset.models = SkyModel(spectral_model=PLmodel, name="Crab")

    estimator = ParameterEstimator(datasets, n_scan_values=10)

    result = estimator.run(PLmodel.amplitude, steps="all")

    assert_allclose(result['value'], 5.142843823441639e-11, rtol=1e-4)
    assert_allclose(result['err'], 6.42467002840316e-12, rtol=1e-4)
    assert_allclose(result['delta_ts'], 353.2092043652601, rtol=1e-4)
    assert_allclose(result['errp'], 6.703e-12, rtol=1e-3)
    assert_allclose(result['errn'], 6.152e-12, rtol=1e-3)
    # Add test for scan
    assert_allclose(result['value_scan'].shape, 10)

@requires_data()
def test_parameter_estimator_3d(crab_datasets_fermi):
    datasets = crab_datasets_fermi
    parameter = datasets[0].models.parameters['amplitude']
    estimator = ParameterEstimator(datasets)

    result = estimator.run(parameter, steps="ts")

    assert_allclose(result['value'], 0.3415434439879935, rtol=1e-4)
    assert_allclose(result['err'], 0.094636191457666734, rtol=1e-4)
    assert_allclose(result['delta_ts'], 13005.938752715907, rtol=1e-4)

@requires_data()
def test_parameter_estimator_3d_no_reoptimization(crab_datasets_fermi):
    datasets = crab_datasets_fermi
    parameter = datasets[0].models.parameters['amplitude']
    estimator = ParameterEstimator(datasets, reoptimize=False, n_scan_values=10)

    result = estimator.run(parameter, steps="all")

    assert datasets[0].models.parameters['alpha'].frozen == False
    assert_allclose(result['value'], 0.3415441642696537, rtol=1e-4)
    assert_allclose(result['err'], 0.006822002813692493, rtol=1e-4)
    assert_allclose(result['delta_ts'], 29695.756280904432, rtol=1e-4)
    assert_allclose(result['errp'], 0.006866385366617123, rtol=1e-3)
    assert_allclose(result['errn'], 0.0067786053845724685, rtol=1e-3)
    assert_allclose(result['ul'], 0.3553685056677599, rtol=1e-3)
    assert_allclose(result['value_scan'].shape, 10)
    assert_allclose(result['value_scan'][0], 0.32107816, atol=1e-3)
    assert_allclose(result['stat_scan'][0], -14874.37451939, atol=1e-3)


