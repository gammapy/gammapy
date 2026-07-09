# Licensed under a 3-clause BSD style license - see LICENSE.rst
from numpy.testing import assert_allclose
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from gammapy.datasets import MapDataset, Datasets
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
import pytest


@requires_data()
class TestEnergyDependentEstimator:
    @pytest.fixture(params=["single", "multiple"], scope="class")
    def estimator_result(self, request):
        stacked_dataset = MapDataset.read(
            "$GAMMAPY_DATA/estimators/mock_DL4/dataset_energy_dependent.fits.gz"
        )

        if request.param == "single":
            datasets = stacked_dataset
        else:
            datasets = Datasets(
                [stacked_dataset, stacked_dataset.copy(name="dataset_copy")]
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

        datasets.models = model
        estimator = EnergyDependentMorphologyEstimator(
            energy_edges=energy_edges, source="source"
        )
        result = estimator.run(datasets)

        return request.param, result

    def test_edep(self, estimator_result):
        mode, result = estimator_result
        results_edep = result["energy_dependence"]["result"]

        assert_allclose(
            results_edep["lon_0"],
            [5.606, 5.608, 5.597] * u.deg,
            atol=1e-2,
        )

        assert_allclose(
            results_edep["lat_0"],
            [0.20289353, 0.20589559, 0.18106776] * u.deg,
            atol=1e-2,
        )

        assert_allclose(
            results_edep["sigma"],
            [0.217, 0.231, 0.135] * u.deg,
            atol=1e-2,
        )

        if mode == "single":
            assert_allclose(result["energy_dependence"]["delta_ts"], 50.719, rtol=1e-3)
            assert_allclose(
                results_edep["sigma_err"],
                [0.0059, 0.0065, 0.0089] * u.deg,
                atol=1e-3,
            )
            assert_allclose(
                results_edep["lat_0_err"],
                [0.00849, 0.0095, 0.014] * u.deg,
                atol=1e-3,
            )
            assert_allclose(
                results_edep["lon_0_err"],
                [0.0084, 0.0095, 0.0147] * u.deg,
                atol=1e-3,
            )
        elif mode == "multiple":
            assert_allclose(result["energy_dependence"]["delta_ts"], 101.738, rtol=1e-3)
            assert_allclose(
                results_edep["sigma_err"],
                [0.0042, 0.003, 0.006] * u.deg,
                atol=2e-3,
            )
            assert_allclose(
                results_edep["lat_0_err"],
                [0.00601639, 0.00678308, 0.01053394] * u.deg,
                atol=1e-3,
            )
            assert_allclose(
                results_edep["lon_0_err"],
                [0.0060174, 0.00664618, 0.0103996] * u.deg,
                atol=1e-3,
            )

    def test_significance(self, estimator_result):
        mode, result = estimator_result
        results_src = result["src_above_bkg"]
        if mode == "single":
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
        elif mode == "multiple":
            assert_allclose(
                results_src["delta_ts"],
                [3362.55821, 579.631],
                rtol=1e-3,
            )
            assert_allclose(
                results_src["significance"],
                [np.inf, 23.5882],
                rtol=1e-3,
            )

    def test_chi2(self, estimator_result):
        mode, result = estimator_result
        results_edep = result["energy_dependence"]["result"]

        chi2_sigma = weighted_chi2_parameter(
            results_edep, parameters=["sigma", "lat_0", "lon_0"]
        )
        if mode == "single":
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
        elif mode == "multiple":
            assert_allclose(
                chi2_sigma["chi2"],
                [180, 3.9, 0.8],
                rtol=3e-1,
            )

            assert_allclose(
                chi2_sigma["significance"],
                [13, 2, 0.9],
                rtol=2e-1,
            )
