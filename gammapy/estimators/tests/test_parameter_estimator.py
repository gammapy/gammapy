# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from gammapy.maps import Map, MapAxis, RegionNDMap
from gammapy.datasets import Datasets, SpectrumDatasetOnOff
from gammapy.estimators.parameter import ParameterEstimator
from gammapy.modeling.models import PowerLawSpectralModel, SkyModel
from gammapy.utils.testing import requires_data


@pytest.fixture
def crab_datasets_1d():
    filename = "$GAMMAPY_DATA/joint-crab/spectra/hess/pha_obs23523.fits"
    dataset = SpectrumDatasetOnOff.read(filename)
    datasets = Datasets([dataset])
    return datasets


@pytest.fixture
def pwl_model():
    return PowerLawSpectralModel(amplitude="3e-11 cm-2s-1TeV-1", index=2.7)


@pytest.fixture
def crab_datasets_fermi():
    filename = "$GAMMAPY_DATA/fermi-3fhl-crab/Fermi-LAT-3FHL_datasets.yaml"
    filename_models = "$GAMMAPY_DATA/fermi-3fhl-crab/Fermi-LAT-3FHL_models.yaml"

    return Datasets.read(filename=filename, filename_models=filename_models)


@requires_data()
def test_parameter_estimator_1d(crab_datasets_1d, pwl_model):
    datasets = crab_datasets_1d

    model = SkyModel(spectral_model=pwl_model, name="Crab")
    model.spectral_model.amplitude.scan_n_values = 10

    for dataset in datasets:
        dataset.models = model

    estimator = ParameterEstimator(selection_optional="all")

    result = estimator.run(datasets, parameter="amplitude")

    assert_allclose(result["amplitude"], 5.1428e-11, rtol=1e-3)
    assert_allclose(result["amplitude_err"], 6.42467e-12, rtol=1e-3)
    assert_allclose(result["ts"], 353.2092, rtol=1e-3)
    assert_allclose(result["stat"], 38.3435, rtol=1e-3)
    assert_allclose(result["stat_null"], 391.5527, rtol=1e-3)
    assert_allclose(result["amplitude_errp"], 6.703e-12, rtol=5e-3)
    assert_allclose(result["amplitude_errn"], 6.152e-12, rtol=5e-3)

    # Add test for scan
    assert_allclose(result["amplitude_scan"].shape, 10)


@requires_data()
def test_parameter_estimator_3d_no_reoptimization(crab_datasets_fermi):
    datasets = crab_datasets_fermi
    parameter = datasets[0].models.parameters["amplitude"]
    parameter.scan_n_values = 10

    estimator = ParameterEstimator(reoptimize=False, selection_optional=["scan"])
    alpha_value = datasets[0].models.parameters["alpha"].value

    result = estimator.run(datasets, parameter)

    assert not datasets[0].models.parameters["alpha"].frozen
    assert_allclose(datasets[0].models.parameters["alpha"].value, alpha_value)
    assert_allclose(result["amplitude"], 0.018251, rtol=1e-3)
    assert_allclose(result["amplitude_scan"].shape, 10)
    assert_allclose(result["amplitude_scan"][0], 0.017282, atol=1e-3)


@requires_data()
def test_parameter_estimator_no_data(crab_datasets_1d, pwl_model):
    datasets = crab_datasets_1d

    model = SkyModel(spectral_model=pwl_model, name="Crab")
    model.spectral_model.amplitude.scan_n_values = 10

    for dataset in datasets:
        dataset.mask_safe.data[...] = False
        dataset.models = model

    estimator = ParameterEstimator(selection_optional="all")

    result = estimator.run(datasets, parameter="amplitude")

    assert np.isnan(result["amplitude"])
    assert np.isnan(result["amplitude_err"])
    assert np.isnan(result["amplitude_errp"])
    assert np.isnan(result["amplitude_errn"])
    assert np.isnan(result["amplitude_ul"])
    assert np.isnan(result["ts"])
    assert np.isnan(result["npred"])
    assert_allclose(result["counts"], 0)

    # Add test for scan
    assert_allclose(result["amplitude_scan"].shape, 10)
    assert np.all(np.isnan(result["stat_scan"]))


def test_parameter_sensitivity_estimator_negative_excess():
    """Test for issue #6719"""
    energy_axis = MapAxis.from_energy_bounds("0.1 TeV", "10 TeV", nbin=1, name="energy")

    dataset_onoff = SpectrumDatasetOnOff.create(
        geom=RegionNDMap.create(
            region="icrs;circle(0, 0, 0.1)", axes=[energy_axis]
        ).geom,
    )
    dataset_onoff.exposure.data[...] = 1e10
    dataset_onoff.acceptance.data[...] = 1
    dataset_onoff.acceptance_off.data[...] = 5
    dataset_onoff.mask_safe.data[...] = True
    model = SkyModel(
        spectral_model=PowerLawSpectralModel(index=2, amplitude="1e-30 cm-2 s-1 TeV-1"),
        name="src",
    )
    dataset_onoff.models = model

    npred_background = Map.from_geom(
        dataset_onoff._geom, data=3000.0
    )  # Do NOT change seed
    dataset_onoff.fake(npred_background=npred_background, random_state=4)

    assert dataset_onoff.excess.data.ravel() < 0

    amplitude = dataset_onoff.models.parameters["amplitude"]

    amplitude.value = 1e-12
    amplitude.error = 1e-20

    estimator = ParameterEstimator(selection_optional=["sensitivity"])

    sensitivity = estimator.estimate_sensitivity(Datasets([dataset_onoff]), amplitude)
    assert_allclose(sensitivity["amplitude_sensitivity"], 9.703913e-14, rtol=5e-3)
