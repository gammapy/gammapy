import pytest
from numpy.testing import assert_allclose
from astropy import units as u
from astropy.coordinates import SkyCoord
from gammapy.datasets import Datasets, MapDataset
from gammapy.estimators.energydependence import (
    EnergyDependenceEstimator,
    weighted_chi2_parameter,
)
from gammapy.modeling.models import (
    GaussianSpatialModel,
    PowerLawSpectralModel,
    SkyModel,
)


@pytest.fixture()
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


class TestEnergyDependentEstimator:
    def __init__(self):
        energy_edges = [1, 3, 5, 20] * u.TeV

        stacked_dataset = MapDataset.read(
            "$GAMMAPY_DATA/estimators/mock_data/dataset_energy_dependent.fits.gz"
        )
        datasets = Datasets([stacked_dataset])
        datasets.models = create_model()

        estimator = EnergyDependenceEstimator(
            energy_edges=energy_edges, source="source"
        )
        self.results = estimator.run(datasets)

    def test_edep(self):
        results_edep = self.results["energy_dependence"]["result"]
        assert_allclose(
            results_edep["lon_0"],
            [5.621402, 5.614581, 5.618528, 5.652494] * u.deg,
            atol=1e-5,
        )
        assert_allclose(
            results_edep["lat_0"],
            [0.20073398, 0.21676375, 0.19342063, 0.20307733] * u.deg,
            atol=1e-5,
        )
        assert_allclose(
            results_edep["sigma"],
            [0.24192075, 0.27437636, 0.2260173, 0.16186429] * u.deg,
            atol=1e-5,
        )
        assert_allclose(
            self.results["energy_dependence"]["delta_ts"], 40.97162512721843, atol=1e-5
        )

    def test_significance(self):
        results_src = self.results["src_above_bkg"]
        assert_allclose(
            results_src["delta_ts"],
            [967.2115532823664, 643.2234423905393, 205.57393366298493],
            atol=1e-5,
        )
        assert_allclose(
            results_src["significance"],
            [30.782094269119664, 24.995567112486306, 13.80497201427469],
            atol=1e-5,
        )

    def test_chi2(self):
        results_edep = self.results["energy_dependence"]["result"]
        chi2_sigma = weighted_chi2_parameter(
            results_edep, parameters=["sigma", "lat_0", "lon_0"]
        )

        assert_allclose(
            chi2_sigma["chi2"],
            [49.90825463240661, 1.25911191815264, 2.4515685629074206],
            atol=1e-5,
        )

        assert_allclose(
            chi2_sigma["significance"],
            [6.752420644157449, 0.623694332110968, 1.0504148916734348],
            atol=1e-5,
        )
