from numpy.testing import assert_allclose
import numpy as np
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
from gammapy.utils.testing import requires_data


@requires_data()
class TestEnergyDependentEstimator:
    def setup_class(self):
        stacked_dataset = MapDataset.read(
            "$GAMMAPY_DATA/estimators/mock_DL4/dataset_energy_dependent.fits.gz"
        )
        energy_edges = [1, 5, 20] * u.TeV

        source_pos = SkyCoord(5.58, 0.2, unit="deg", frame="galactic")

        spectral_model = PowerLawSpectralModel(
            index=2.94,
            amplitude=9.8e-12 * u.Unit("cm-2 s-1 TeV-1"),
            reference=1.0 * u.TeV,
        )
        spatial_model = GaussianSpatialModel(
            lon_0=source_pos.l, lat_0=source_pos.b, frame="galactic", sigma=0.2 * u.deg
        )

        model = SkyModel(
            spatial_model=spatial_model, spectral_model=spectral_model, name="source"
        )

        model.spectral_model.index.frozen = True

        model.spatial_model.lon_0.min = source_pos.galactic.l.deg - 0.8
        model.spatial_model.lon_0.max = source_pos.galactic.l.deg + 0.8
        model.spatial_model.lat_0.min = source_pos.galactic.b.deg - 0.8
        model.spatial_model.lat_0.max = source_pos.galactic.b.deg + 0.8

        stacked_dataset.models = model
        estimator = EnergyDependentMorphologyEstimator(
            energy_edges=energy_edges, source="source"
        )
        self.estimator_result = estimator.run(stacked_dataset)

    def test_edep(self):
        results_edep = self.estimator_result["energy_dependence"]["result"]
        assert_allclose(
            results_edep["lon_0"],
            [5.606467, 5.608664, 5.597394] * u.deg,
            rtol=1e-3,
        )
        assert_allclose(
            results_edep["lat_0"],
            [0.20289353, 0.20589559, 0.18106776] * u.deg,
            rtol=1e-3,
        )
        assert_allclose(
            results_edep["sigma"],
            [0.21709024, 0.2315993, 0.13505759] * u.deg,
            rtol=1e-3,
        )
        assert_allclose(
            self.estimator_result["energy_dependence"]["delta_ts"], 50.719, rtol=1e-3
        )

    def test_significance(self):
        results_src = self.estimator_result["src_above_bkg"]
        assert_allclose(
            results_src["delta_ts"],
            [1683.1128, 289.8156],
            rtol=1e-3,
        )
        assert_allclose(
            results_src["significance"],
            [np.inf, 16.546],
            rtol=1e-3,
        )

    def test_chi2(self):
        results_edep = self.estimator_result["energy_dependence"]["result"]
        chi2_sigma = weighted_chi2_parameter(
            results_edep, parameters=["sigma", "lat_0", "lon_0"]
        )

        assert_allclose(
            chi2_sigma["chi2"],
            [75.735899, 1.962729, 0.413059],
            rtol=1e-2,
        )

        assert_allclose(
            chi2_sigma["significance"],
            [8.7026, 1.4009, 0.64269],
            rtol=1e-2,
        )
