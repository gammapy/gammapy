# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
import astropy.units as u
from gammapy.datasets import Datasets, SpectrumDatasetOnOff
from gammapy.estimators.flux import FluxEstimator
from gammapy.modeling.models import PowerLawSpectralModel, SkyModel
from gammapy.utils.testing import requires_data, requires_dependency


@pytest.fixture(scope="session")
def fermi_datasets():
    fermi_datasets = Datasets.read(
        "$GAMMAPY_DATA/fermi-3fhl-crab",
        "Fermi-LAT-3FHL_datasets.yaml",
        "Fermi-LAT-3FHL_models.yaml",
    )
    return fermi_datasets


@pytest.fixture(scope="session")
def hess_datasets():
    datasets = Datasets([])
    pwl = PowerLawSpectralModel(amplitude="3.5e-11 cm-2s-1TeV-1", index=2.7)
    model = SkyModel(spectral_model=pwl, name="Crab")

    for obsid in [23523, 23526]:
        dataset = SpectrumDatasetOnOff.from_ogip_files(
            f"$GAMMAPY_DATA/joint-crab/spectra/hess/pha_obs{obsid}.fits"
        )
        dataset.models = model
        datasets.append(dataset)

    return datasets


@requires_data()
@requires_dependency("iminuit")
def test_flux_estimator_fermi_no_reoptimization(fermi_datasets):
    estimator = FluxEstimator(
        0,
        energy_range=["1 GeV", "100 GeV"],
        norm_n_values=5,
        norm_min=0.5,
        norm_max=2,
        reoptimize=False,
    )
    result = estimator.run(fermi_datasets)

    assert_allclose(result["norm"], 1.012374, atol=1e-3)
    assert_allclose(result["ts"], 28086.66085, atol=1e-3)
    assert_allclose(result["norm_err"], 0.01998, atol=1e-3)
    assert_allclose(result["norm_errn"], 0.0199, atol=1e-3)
    assert_allclose(result["norm_errp"], 0.0199, atol=1e-3)
    assert len(result["norm_scan"]) == 5
    assert_allclose(result["norm_scan"][0], 0.5)
    assert_allclose(result["norm_scan"][-1], 2)


@requires_data()
@requires_dependency("iminuit")
def test_flux_estimator_fermi_with_reoptimization(fermi_datasets):
    estimator = FluxEstimator(0, energy_range=["1 GeV", "100 GeV"], reoptimize=True)
    result = estimator.run(fermi_datasets, steps=["err", "ts"])

    assert_allclose(result["norm"], 1.012374, atol=1e-3)
    assert_allclose(result["ts"], 20896.27951, atol=1e-3)
    assert_allclose(result["norm_err"], 0.01998, atol=1e-3)


@requires_data()
@requires_dependency("iminuit")
def test_flux_estimator_1d(hess_datasets):
    estimator = FluxEstimator(source="Crab", energy_range=[1, 10] * u.TeV)
    result = estimator.run(hess_datasets, steps=["err", "ts", "errp-errn", "ul"])

    assert_allclose(result["norm"], 1.176789, atol=1e-3)
    assert_allclose(result["ts"], 693.111777, atol=1e-3)
    assert_allclose(result["norm_err"], 0.078087, atol=1e-3)
    assert_allclose(result["norm_errn"], 0.078046, atol=1e-3)
    assert_allclose(result["norm_errp"], 0.081665, atol=1e-3)
    assert_allclose(result["norm_ul"], 1.431722, atol=1e-3)


@requires_data()
def test_inhomogeneous_datasets(fermi_datasets, hess_datasets):
    fermi_datasets.append(hess_datasets[0])
    with pytest.raises(ValueError):
        estimator = FluxEstimator(source=0, energy_range=[1, 10] * u.TeV)
        estimator.run(fermi_datasets)


@requires_data()
def test_flux_estimator_incorrect_energy_range():
    with pytest.raises(ValueError):
        FluxEstimator(source="Crab", energy_range=[1, 3, 10] * u.TeV)
    with pytest.raises(ValueError):
        FluxEstimator(source="Crab", energy_range=[10, 1] * u.TeV)
