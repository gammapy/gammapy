# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Tests for FluxCollectionEstimator
"""

import numpy as np
import astropy.units as u
import pytest


from gammapy.maps import Map, WcsGeom, MapAxis
from gammapy.modeling import Fit, Sampler
from gammapy.modeling.models import (
    Models,
    SkyModel,
    PowerLawSpectralModel,
    FoVBackgroundModel,
)
from gammapy.datasets import MapDataset, Datasets

from gammapy.estimators.points.sed import FluxCollectionEstimator


@pytest.fixture
def simple_dataset():
    axis = MapAxis.from_energy_bounds(0.1, 10, 1, unit="TeV")
    geom = WcsGeom.create(npix=20, binsz=0.02, axes=[axis])
    simple_dataset = MapDataset.create(geom)
    simple_dataset.mask_safe += np.ones(simple_dataset.data_shape, dtype=bool)
    simple_dataset.counts += 2
    simple_dataset.background += 1

    spectral = PowerLawSpectralModel(index=2.0, amplitude="1e-12 cm-2 s-1 TeV-1")
    simple_dataset.models = [
        SkyModel(spectral_model=spectral, name="test-source"),
        FoVBackgroundModel(dataset_name=simple_dataset.name),
    ]

    return simple_dataset


@pytest.fixture
def energy_edges():
    return np.array([1.0, 3.0, 10.0]) * u.TeV


class MockFit(Fit):
    """Mock Fit returning fixed parameter results."""

    def run(self, datasets):
        # Set norm = 2, error = 0.2 for all spectral models with a 'norm'
        for ds in datasets:
            for ev in ds.evaluators.values():
                m = ev.model
                if hasattr(m.spectral_model, "norm"):
                    m.spectral_model.norm.value = 2.0
                    m.spectral_model.norm.error = 0.2

        return {"success": True}

    def confidence(self, datasets, parameter, sigma):
        return {"errn": 0.1, "errp": 0.3}


class MockFitMulti(Fit):
    """Mock Fit for multiple models with distinct values."""

    def run(self, datasets):
        # assign distinct best-fit norms for each model i
        for ds in datasets:
            for i, ev in enumerate(ds.evaluators.values()):
                m = ev.model
                if hasattr(m.spectral_model, "norm"):
                    m.spectral_model.norm.value = 1.0 + i
                    m.spectral_model.norm.error = 0.1 + 0.05 * i
        return {"success": True}

    def confidence(self, datasets, parameter, sigma):
        return {"errn": 0.1 * sigma, "errp": 0.2 * sigma}


@pytest.fixture
def mock_fit():
    return MockFit()


@pytest.fixture
def mock_fit_multi():
    return MockFitMulti()


class MockSamplerResult:
    """Container for sampler_results for ns parameters."""

    def __init__(self, ns):
        n_samples = 100
        points = 2.0 + 0.1 * np.random.randn(n_samples, ns)
        weights = np.ones(n_samples) / n_samples

        self.sampler_results = {
            "weighted_samples": {
                "points": points,
                "weights": weights,
            }
        }


class MockSampler(Sampler):
    """Mock sampler for single-source."""

    def __init__(self, ns=1):
        super().__init__(backend="mock", sampler_opts={})
        self.ns = ns

    def run(self, datasets):
        return MockSamplerResult(self.ns)


@pytest.fixture
def mock_sampler():
    return MockSampler(ns=1)


class MockSamplerResultMulti:
    def __init__(self, ns):
        n_samples = 200
        # points: shape (n_samples, ns)
        points = 1.0 + np.arange(ns) * 0.5 + 0.1 * np.random.randn(n_samples, ns)
        weights = np.ones(n_samples) / n_samples

        self.sampler_results = {
            "weighted_samples": {
                "points": points,
                "weights": weights,
            }
        }


class MockSamplerMulti(Sampler):
    """Mock sampler for multiple-source."""

    def __init__(self, ns=1):
        super().__init__(backend="mock", sampler_opts={})
        self.ns = ns

    def run(self, datasets):
        return MockSamplerResultMulti(self.ns)


@pytest.fixture
def mock_sampler_multi():
    return MockSamplerMulti(ns=2)


def test_init(simple_dataset, energy_edges):
    model = simple_dataset.models["test-source"]
    est = FluxCollectionEstimator(
        energy_edges=energy_edges,
        models=[model],
    )
    assert isinstance(est.solver, Sampler)


def test_run(simple_dataset, energy_edges, mock_fit):
    model = simple_dataset.models["test-source"]
    est = FluxCollectionEstimator(
        energy_edges=energy_edges,
        models=[model],
        solver=mock_fit,
        reoptimize=True,
    )

    result = est.run(Datasets([simple_dataset]))
    fp = result["flux_points"]["test-source"]

    assert len(fp["dnde"].data) == 2
    assert fp["dnde"].unit == u.Unit("cm-2 s-1 TeV-1")
    assert np.all(np.isfinite(fp["dnde"]))
    assert np.all(fp["ts"] >= 0)


def test_asymmetric_errors_present(simple_dataset, energy_edges, mock_fit):
    model = simple_dataset.models["test-source"]
    est = FluxCollectionEstimator(
        energy_edges=energy_edges,
        models=[model],
        solver=mock_fit,
        selection_optional=["errn-errp"],
    )

    result = est.run(Datasets([simple_dataset]))
    fp = result["flux_points"]["test-source"]
    assert np.all(np.isfinite(fp["dnde_errn"]))
    assert np.all(np.isfinite(fp["dnde_errp"]))


def test_inconsistent_geometry_raises(simple_dataset):
    # dataset with different geom
    geom2 = WcsGeom.create(npix=(3, 3), binsz=0.1)
    ds2 = simple_dataset.copy(name="bad-ds")
    ds2.counts = Map.from_geom(geom2, data=np.ones(geom2.data_shape))

    model = simple_dataset.models["test-source"]
    est = FluxCollectionEstimator(
        energy_edges=[1, 3, 10] * u.TeV,
        models=[model],
        solver=MockFit(),
    )

    with pytest.raises(ValueError):
        est.run(Datasets([simple_dataset, ds2]))


def test_run_sampler_case(simple_dataset, energy_edges, mock_sampler):
    model = simple_dataset.models["test-source"]

    est = FluxCollectionEstimator(
        energy_edges=energy_edges,
        models=[model],
        solver=mock_sampler,
    )

    result = est.run(Datasets([simple_dataset]))
    fp = result["flux_points"]["test-source"]

    assert len(fp["dnde"].data) == 2
    assert np.all(np.isfinite(fp["dnde"]))
    assert np.all(np.isfinite(fp["dnde_errn"]))
    assert np.all(np.isfinite(fp["dnde_errp"]))
    assert np.all(np.isfinite(fp["dnde_ul"]))
    assert np.all(np.isfinite(fp["ts"]))

    # sampler-only data
    samples = result["samples"]["dnde"]
    assert "test-source" in samples
    assert len(samples["test-source"]) == 2


def test_run_sampler_multi_source(simple_dataset, energy_edges, mock_sampler_multi):
    # Add two sources
    m1 = simple_dataset.models["test-source"]
    m2 = m1.copy(name="test-source-2")
    simple_dataset.models = Models([m1, m2])

    est = FluxCollectionEstimator(
        energy_edges=energy_edges,
        models=[m1, m2],
        solver=mock_sampler_multi,
    )

    result = est.run(Datasets([simple_dataset]))
    flux_points = result["flux_points"]
    nbin = len(energy_edges) - 1

    assert set(flux_points.keys()) == {"test-source", "test-source-2"}

    for name in ["test-source", "test-source-2"]:
        fp = flux_points[name]
        assert len(fp["dnde"].data) == nbin
        assert np.all(np.isfinite(fp["dnde"]))
        assert np.all(np.isfinite(fp["dnde_errn"]))
        assert np.all(np.isfinite(fp["dnde_errp"]))
        assert np.all(np.isfinite(fp["dnde_ul"]))
        assert np.all(np.isfinite(fp["ts"]))

    samples = result["samples"]["dnde"]
    assert "test-source" in samples
    assert "test-source-2" in samples
    assert len(samples["test-source"]) == nbin
    assert samples["test-source"][0].shape[0] == 200


def test_run_fit_multi_source(simple_dataset, energy_edges, mock_fit_multi):
    m1 = simple_dataset.models["test-source"]
    m2 = m1.copy(name="test-source-2")
    simple_dataset.models = Models([m1, m2])

    est = FluxCollectionEstimator(
        energy_edges=energy_edges,
        models=[m1, m2],
        solver=mock_fit_multi,
        selection_optional=["errn-errp"],
    )

    result = est.run(Datasets([simple_dataset]))
    flux_points = result["flux_points"]
    nbin = len(energy_edges) - 1

    assert set(flux_points.keys()) == {"test-source", "test-source-2"}

    for name in ["test-source", "test-source-2"]:
        fp = flux_points[name]
        assert len(fp["dnde"].data) == nbin
        assert np.all(np.isfinite(fp["dnde"]))
        assert np.all(np.isfinite(fp["dnde_errn"]))
        assert np.all(np.isfinite(fp["dnde_errp"]))
        assert np.all(np.isfinite(fp["dnde_ul"]))
        assert np.all(fp["ts"] >= 0)

    assert result["solver_results"].shape == (nbin,)


def test_run_multi_dataset(simple_dataset, energy_edges, mock_fit_multi):
    ds2 = simple_dataset.copy(name="dataset-2")
    ds2.counts = ds2.counts.copy(data=12 * np.ones(ds2.counts.data.shape))
    ds2.background = ds2.background.copy(data=9 * np.ones(ds2.background.data.shape))

    datasets = Datasets([simple_dataset, ds2])

    m1 = simple_dataset.models["test-source"]
    m2 = m1.copy(name="test-source-2")

    models = Models([m1, m2])
    simple_dataset.models = models
    ds2.models = models

    est = FluxCollectionEstimator(
        energy_edges=energy_edges,
        models=[m1, m2],
        solver=mock_fit_multi,
        selection_optional=["errn-errp"],
        reoptimize=True,
    )

    result = est.run(datasets)
    flux_points = result["flux_points"]
    nbin = len(energy_edges) - 1

    for name in ["test-source", "test-source-2"]:
        fp = flux_points[name]
        assert len(fp["dnde"].data) == nbin
        assert np.all(np.isfinite(fp["dnde"]))
        assert np.all(np.isfinite(fp["ts"]))
