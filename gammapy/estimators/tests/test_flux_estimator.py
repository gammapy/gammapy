# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
from gammapy.datasets import Datasets, SpectrumDatasetOnOff
from gammapy.modeling.models import SkyModel, PowerLawSpectralModel
from gammapy.estimators import FluxEstimator
from gammapy.utils.testing import requires_data, requires_dependency


@pytest.fixture(scope="session")
def fermi_datasets():
    fermi_datasets = Datasets.read("$GAMMAPY_DATA/fermi-3fhl-crab/Fermi-LAT-3FHL_datasets.yaml",
                                   "$GAMMAPY_DATA/fermi-3fhl-crab/Fermi-LAT-3FHL_models.yaml")
    return fermi_datasets

@pytest.fixture(scope="session")
def hess_datasets():
    datasets = Datasets([])
    for obsid in [23523, 23526]:
        datasets.append(
            SpectrumDatasetOnOff.from_ogip_files(f"$GAMMAPY_DATA/joint-crab/spectra/hess/pha_obs{obsid}.fits")
        )
    PLmodel = PowerLawSpectralModel(amplitude="3.5e-11 cm-2s-1TeV-1", index=2.7)
    for dataset in datasets:
        dataset.models = SkyModel(spectral_model=PLmodel, name="Crab")
    return datasets

@requires_data()
@requires_dependency("iminuit")
def test_flux_estimator_fermi_no_reoptimization(fermi_datasets):
    estimator = FluxEstimator(fermi_datasets, norm_n_values=5, norm_min=0.5, norm_max=2, reoptimize=False)
    result = estimator.run("1 GeV", "100 GeV")

    assert_allclose(result["norm"], 1.00, atol=1e-3)
    assert_allclose(result["delta_ts"], 29695.756278, atol=1e-3)
    assert_allclose(result["err"], 0.01998, atol=1e-3)
    assert_allclose(result["errn"], 0.0199, atol=1e-3)
    assert_allclose(result["errp"], 0.0199, atol=1e-3)
    assert len(result["norm_scan"]) == 5
    assert_allclose(result["norm_scan"][0], 0.5)
    assert_allclose(result["norm_scan"][-1], 2)

@requires_data()
@requires_dependency("iminuit")
def test_flux_estimator_fermi_with_reoptimization(fermi_datasets):
    estimator = FluxEstimator(fermi_datasets, reoptimize=True)
    result = estimator.run("1 GeV", "100 GeV", steps=["ts"])

    print(estimator)
    assert_allclose(result["norm"], 1.00, atol=1e-3)
    assert_allclose(result["delta_ts"], 13005.938759, atol=1e-3)
    assert_allclose(result["err"], 0.01998, atol=1e-3)

@requires_data()
@requires_dependency("iminuit")
def test_flux_estimator_1d(hess_datasets):
    estimator = FluxEstimator(hess_datasets, source="Crab")
    result = estimator.run('1 TeV', '10 TeV', steps=["ts", "errp-errn", "ul"])

    assert_allclose(result["norm"], 1.176789, atol=1e-3)
    assert_allclose(result["delta_ts"], 693.111777, atol=1e-3)
    assert_allclose(result["err"], 0.079840, atol=1e-3)
    assert_allclose(result["errn"], 0.078046, atol=1e-3)
    assert_allclose(result["errp"], 0.081665, atol=1e-3)
    assert_allclose(result["ul"], 1.431722, atol=1e-3)

@requires_data()
def test_inhomogeneous_datasets(fermi_datasets, hess_datasets):
    fermi_datasets.append(hess_datasets[0])
    with pytest.raises(ValueError):
        FluxEstimator(fermi_datasets, source=0)
