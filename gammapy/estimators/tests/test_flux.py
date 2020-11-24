# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from gammapy.datasets import Datasets, SpectrumDatasetOnOff
from gammapy.estimators.flux import FluxEstimator
from gammapy.modeling.models import PowerLawSpectralModel, SkyModel
from gammapy.utils.testing import requires_data, requires_dependency


@pytest.fixture(scope="session")
def fermi_datasets():
    filename = "$GAMMAPY_DATA/fermi-3fhl-crab/Fermi-LAT-3FHL_datasets.yaml"
    filename_models = "$GAMMAPY_DATA/fermi-3fhl-crab/Fermi-LAT-3FHL_models.yaml"

    return Datasets.read(filename=filename, filename_models=filename_models)


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
        energy_min="1 GeV",
        energy_max="100 GeV",
        norm_n_values=5,
        norm_min=0.5,
        norm_max=2,
        reoptimize=False,
    )

    result = estimator.run(fermi_datasets)

    assert_allclose(result["norm"], 0.98949, atol=1e-3)
    assert_allclose(result["ts"], 25082.190245, atol=1e-3)
    assert_allclose(result["norm_err"], 0.01998, atol=1e-3)
    assert_allclose(result["norm_errn"], 0.0199, atol=1e-3)
    assert_allclose(result["norm_errp"], 0.0199, atol=1e-3)
    assert len(result["norm_scan"]) == 5
    assert_allclose(result["norm_scan"][0], 0.5)
    assert_allclose(result["norm_scan"][-1], 2)
    assert_allclose(result["e_min"], 10 * u.GeV, atol=1e-3)
    assert_allclose(result["e_max"], 83.255 * u.GeV, atol=1e-3)


@requires_data()
@requires_dependency("iminuit")
def test_flux_estimator_fermi_with_reoptimization(fermi_datasets):
    estimator = FluxEstimator(
        0,
        energy_min="1 GeV",
        energy_max="100 GeV",
        reoptimize=True,
        selection_optional=None,
    )
    result = estimator.run(fermi_datasets)

    assert_allclose(result["norm"], 0.989989, atol=1e-3)
    assert_allclose(result["ts"], 25082.190245, atol=1e-3)
    assert_allclose(result["norm_err"], 0.01998, atol=1e-3)


@requires_data()
@requires_dependency("iminuit")
def test_flux_estimator_1d(hess_datasets):
    estimator = FluxEstimator(
        source="Crab",
        energy_min=1 * u.TeV,
        energy_max=10 * u.TeV,
        selection_optional=["errn-errp", "ul"],
    )
    result = estimator.run(hess_datasets)

    assert_allclose(result["norm"], 1.218139, atol=1e-3)
    assert_allclose(result["ts"], 527.492959, atol=1e-3)
    assert_allclose(result["norm_err"], 0.095496, atol=1e-3)
    assert_allclose(result["norm_errn"], 0.093204, atol=1e-3)
    assert_allclose(result["norm_errp"], 0.097818, atol=1e-3)
    assert_allclose(result["norm_ul"], 1.525773, atol=1e-3)
    assert_allclose(result["e_min"], 1 * u.TeV, atol=1e-3)
    assert_allclose(result["e_max"], 10 * u.TeV, atol=1e-3)


@requires_data()
@requires_dependency("iminuit")
def test_flux_estimator_incorrect_energy_range(fermi_datasets):
    with pytest.raises(ValueError):
        FluxEstimator(source="Crab", energy_min=10 * u.TeV, energy_max=1 * u.TeV)

    fe = FluxEstimator(
        source="Crab Nebula", energy_min=0.18 * u.TeV, energy_max=0.2 * u.TeV
    )

    result = fe.run(fermi_datasets)

    assert np.isnan(result["norm"])


@requires_data()
@requires_dependency("iminuit")
def test_inhomogeneous_datasets(fermi_datasets, hess_datasets):

    for dataset in hess_datasets:
        dataset.models = fermi_datasets.models

    datasets = Datasets()

    datasets.extend(fermi_datasets)
    datasets.extend(hess_datasets)

    estimator = FluxEstimator(
        source="Crab Nebula",
        energy_min=1 * u.TeV,
        energy_max=10 * u.TeV,
        selection_optional=None,
    )
    result = estimator.run(datasets)

    assert_allclose(result["norm"], 1.190622, atol=1e-3)
    assert_allclose(result["ts"], 660.422291, atol=1e-3)
    assert_allclose(result["norm_err"], 0.090744, atol=1e-3)
    assert_allclose(result["e_min"], 0.693145 * u.TeV, atol=1e-3)
    assert_allclose(result["e_max"], 2 * u.TeV, atol=1e-3)
