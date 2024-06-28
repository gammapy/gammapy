# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from gammapy.datasets import Datasets, SpectrumDatasetOnOff
from gammapy.estimators.flux import FluxEstimator
from gammapy.modeling import Parameter
from gammapy.modeling.models import (
    Models,
    NaimaSpectralModel,
    PowerLawNormSpectralModel,
    PowerLawSpectralModel,
    SkyModel,
)
from gammapy.utils.testing import requires_data, requires_dependency


@pytest.fixture()
def fermi_datasets():
    from gammapy.datasets import Datasets

    filename = "$GAMMAPY_DATA/fermi-3fhl-crab/Fermi-LAT-3FHL_datasets.yaml"
    filename_models = "$GAMMAPY_DATA/fermi-3fhl-crab/Fermi-LAT-3FHL_models.yaml"
    return Datasets.read(filename=filename, filename_models=filename_models)


@pytest.fixture(scope="session")
def hess_datasets():
    datasets = Datasets()
    pwl = PowerLawSpectralModel(amplitude="3.5e-11 cm-2s-1TeV-1", index=2.7)
    model = SkyModel(spectral_model=pwl, name="Crab")

    for obsid in [23523, 23526]:
        dataset = SpectrumDatasetOnOff.read(
            f"$GAMMAPY_DATA/joint-crab/spectra/hess/pha_obs{obsid}.fits"
        )
        dataset.models = model
        datasets.append(dataset)

    return datasets


@requires_data()
def test_flux_estimator_fermi_no_reoptimization(fermi_datasets):
    norm = Parameter(
        value=1, name="norm", scan_n_values=5, scan_min=0.5, scan_max=2, interp="log"
    )
    estimator = FluxEstimator(
        0,
        norm=norm,
        selection_optional="all",
        reoptimize=False,
    )

    datasets = fermi_datasets.slice_by_energy(energy_min="1 GeV", energy_max="100 GeV")
    datasets.models = fermi_datasets.models

    result = estimator.run(datasets)

    assert_allclose(result["norm"], 0.98949, atol=1e-3)
    assert_allclose(result["ts"], 25083.75408, rtol=1e-3)
    assert_allclose(result["norm_err"], 0.01998, atol=1e-3)
    assert_allclose(result["norm_errn"], 0.0199, atol=1e-3)
    assert_allclose(result["norm_errp"], 0.0199, atol=1e-3)
    assert len(result["norm_scan"]) == 5
    assert_allclose(result["norm_scan"][0], 0.5)
    assert_allclose(result["norm_scan"][-1], 2)
    assert_allclose(result["e_min"], 10 * u.GeV, atol=1e-3)
    assert_allclose(result["e_max"], 83.255 * u.GeV, atol=1e-3)


@requires_data()
def test_flux_estimator_fermi_with_reoptimization(fermi_datasets):
    estimator = FluxEstimator(0, selection_optional=None, reoptimize=True)

    datasets = fermi_datasets.slice_by_energy(energy_min="1 GeV", energy_max="100 GeV")
    datasets.models = fermi_datasets.models

    result = estimator.run(datasets)

    assert_allclose(result["norm"], 0.989989, atol=1e-3)
    assert_allclose(result["ts"], 18729.368105, rtol=1e-3)
    assert_allclose(result["norm_err"], 0.01998, atol=1e-3)


@requires_data()
def test_flux_estimator_1d(hess_datasets):
    estimator = FluxEstimator(
        source="Crab", selection_optional=["errn-errp", "ul"], reoptimize=False
    )
    datasets = hess_datasets.slice_by_energy(
        energy_min=1 * u.TeV,
        energy_max=10 * u.TeV,
    )
    datasets.models = hess_datasets.models

    result = estimator.run(datasets)

    assert_allclose(result["norm"], 1.218139, atol=1e-3)
    assert_allclose(result["ts"], 527.492959, atol=1e-3)
    assert_allclose(result["norm_err"], 0.095496, atol=1e-3)
    assert_allclose(result["norm_errn"], 0.093204, atol=1e-3)
    assert_allclose(result["norm_errp"], 0.097818, atol=1e-3)
    assert_allclose(result["norm_ul"], 1.418475, atol=1e-3)
    assert_allclose(result["e_min"], 1 * u.TeV, atol=1e-3)
    assert_allclose(result["e_max"], 10 * u.TeV, atol=1e-3)
    assert_allclose(result["npred"], [93.209263, 93.667283], atol=1e-3)
    assert_allclose(result["npred_excess"], [86.27813, 88.6715], atol=1e-3)


@requires_data()
def test_inhomogeneous_datasets(fermi_datasets, hess_datasets):
    datasets = Datasets()

    datasets.extend(fermi_datasets)
    datasets.extend(hess_datasets)

    datasets = datasets.slice_by_energy(
        energy_min=1 * u.TeV,
        energy_max=10 * u.TeV,
    )
    datasets.models = fermi_datasets.models

    estimator = FluxEstimator(
        source="Crab Nebula", selection_optional=[], reoptimize=True
    )
    result = estimator.run(datasets)

    assert_allclose(result["norm"], 1.190622, atol=1e-3)
    assert_allclose(result["ts"], 612.503013, atol=1e-3)
    assert_allclose(result["norm_err"], 0.090744, atol=1e-3)
    assert_allclose(result["e_min"], 0.693145 * u.TeV, atol=1e-3)
    assert_allclose(result["e_max"], 10 * u.TeV, atol=1e-3)


def test_flux_estimator_norm_range():
    model = SkyModel.create("pl", "gauss", name="test")

    norm = Parameter(value=1, name="norm", min=1e-3, max=1e2, interp="log")
    estimator = FluxEstimator(
        source="test", norm=norm, selection_optional=[], reoptimize=True
    )

    scale_model = estimator.get_scale_model(Models([model]))

    assert_allclose(scale_model.norm.min, 1e-3)
    assert_allclose(scale_model.norm.max, 1e2)
    assert scale_model.norm.interp == "log"


def test_flux_estimator_norm_dict():
    norm = dict(value=1, name="norm", min=1e-3, max=1e2, interp="log")
    estimator = FluxEstimator(
        source="test", norm=norm, selection_optional=[], reoptimize=True
    )
    assert estimator.norm.value == 1
    assert_allclose(estimator.norm.min, 1e-3)
    assert_allclose(estimator.norm.max, 1e2)
    assert estimator.norm.interp == "log"


def test_flux_estimator_compound_model():
    pl = PowerLawSpectralModel()
    pl.amplitude.min = 1e-15
    pl.amplitude.max = 1e-10

    pln = PowerLawNormSpectralModel()
    pln.norm.value = 0.1
    pln.norm.frozen = True
    spectral_model = pl * pln
    model = SkyModel(spectral_model=spectral_model, name="test")

    norm = Parameter(value=1, name="norm", min=1e-3, max=1e2, interp="log")
    estimator = FluxEstimator(
        source="test", norm=norm, selection_optional=[], reoptimize=True
    )

    scale_model = estimator.get_scale_model(Models([model]))
    assert_allclose(scale_model.norm.min, 1e-3)
    assert_allclose(scale_model.norm.max, 1e2)

    pl2 = PowerLawSpectralModel()
    pl2.amplitude.min = 1e-14
    pl2.amplitude.max = 1e-10
    spectral_model2 = pl + pl2
    model2 = SkyModel(spectral_model=spectral_model2, name="test")

    pl2.amplitude.frozen = True
    scale_model = estimator.get_scale_model(Models([model2]))
    assert_allclose(scale_model.norm.min, 1e-3)

    pl.amplitude.frozen = True
    pl2.amplitude.frozen = False
    scale_model = estimator.get_scale_model(Models([model2]))
    assert_allclose(scale_model.norm.min, 1e-3)

    pl2.amplitude.frozen = True
    scale_model = estimator.get_scale_model(Models([model2]))
    assert_allclose(scale_model.norm.min, 1e-3)


@requires_dependency("naima")
def test_flux_estimator_naima_model():
    import naima

    ECPL = naima.models.ExponentialCutoffPowerLaw(
        1e36 * u.Unit("1/eV"), 1 * u.TeV, 2.1, 13 * u.TeV
    )
    IC = naima.models.InverseCompton(ECPL, seed_photon_fields=["CMB"])
    naima_model = NaimaSpectralModel(IC)

    model = SkyModel(spectral_model=naima_model, name="test")

    estimator = FluxEstimator(source="test", selection_optional=[], reoptimize=True)

    scale_model = estimator.get_scale_model(Models([model]))

    assert_allclose(scale_model.norm.min, np.nan)
    assert_allclose(scale_model.norm.max, np.nan)
