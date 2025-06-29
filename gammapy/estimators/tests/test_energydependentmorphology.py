import pytest
from numpy.testing import assert_allclose
from astropy import units as u
from astropy.coordinates import SkyCoord
from gammapy.datasets import MapDataset
from gammapy.estimators.energydependentmorphology import (
    EnergyDependentMorphologyEstimator,
    weighted_chi2_parameter,
)
from gammapy.modeling.models import (
    GaussianSpatialModel,
    PowerLawSpectralModel,
    SkyModel,
)


@pytest.fixture(scope="module")
def create_model():
    source_pos = SkyCoord(5.58, 0.2, unit="deg", frame="galactic")

    spectral_model = PowerLawSpectralModel(
        index=2.94, amplitude=9.8e-12 * u.Unit("cm-2 s-1 TeV-1"), reference=1.0 * u.TeV
    )
    spatial_model = GaussianSpatialModel(
        lon_0=source_pos.l, lat_0=source_pos.b, frame="galactic", sigma=0.2 * u.deg
    )

    model = SkyModel(
        spatial_model=spatial_model, spectral_model=spectral_model, name="source"
    )

    model.spatial_model.lon_0.frozen = False
    model.spatial_model.lat_0.frozen = False
    model.spatial_model.sigma.frozen = False

    model.spectral_model.amplitude.frozen = False
    model.spectral_model.index.frozen = True

    model.spatial_model.lon_0.min = source_pos.galactic.l.deg - 0.8
    model.spatial_model.lon_0.max = source_pos.galactic.l.deg + 0.8
    model.spatial_model.lat_0.min = source_pos.galactic.b.deg - 0.8
    model.spatial_model.lat_0.max = source_pos.galactic.b.deg + 0.8

    return model


@pytest.fixture(scope="module")
def estimator_result(create_model):
    energy_edges = [1, 3, 5, 20] * u.TeV
    stacked_dataset = MapDataset.read(
        "$GAMMAPY_DATA/estimators/mock_DL4/dataset_energy_dependent.fits.gz"
    )
    stacked_dataset.models = create_model
    estimator = EnergyDependentMorphologyEstimator(
        energy_edges=energy_edges, source="source"
    )
    return estimator.run(stacked_dataset)


def test_edep(estimator_result):
    results_edep = estimator_result["energy_dependence"]["result"]
    assert_allclose(
        results_edep["lon_0"],
        [5.6067162, 5.601791, 5.6180701, 5.5973948] * u.deg,
        rtol=1e-3,
    )
    assert_allclose(
        results_edep["lat_0"],
        [0.20237541, 0.21819575, 0.18371523, 0.18106852] * u.deg,
        rtol=1e-3,
    )
    assert_allclose(
        results_edep["sigma"],
        [0.21563528, 0.25686477, 0.19736596, 0.13505605] * u.deg,
        rtol=1e-3,
    )
    assert_allclose(estimator_result["energy_dependence"]["delta_ts"], 75.62, rtol=1e-3)


def test_significance(estimator_result):
    results_src = estimator_result["src_above_bkg"]
    assert_allclose(
        results_src["delta_ts"],
        [998.0521965029693, 712.8735641098574, 289.81556949490914],
        rtol=1e-3,
    )
    assert_allclose(
        results_src["significance"],
        [31.27752315246094, 26.34612970747113, 16.54625269423397],
        rtol=1e-3,
    )


def test_chi2(estimator_result):
    results_edep = estimator_result["energy_dependence"]["result"]
    chi2_sigma = weighted_chi2_parameter(
        results_edep, parameters=["sigma", "lat_0", "lon_0"]
    )

    assert_allclose(
        chi2_sigma["chi2"],
        [87.84278516393066, 4.605432972153188, 1.320491077667271],
        rtol=1e-3,
    )

    assert_allclose(
        chi2_sigma["significance"],
        [9.107664118611664, 1.6449173252682943, 0.6484028260024965],
        rtol=1e-3,
    )
