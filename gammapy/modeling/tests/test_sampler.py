# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import importlib
from gammapy.utils.testing import requires_data
from numpy.testing import assert_allclose
from gammapy.modeling.models import SkyModel
from gammapy.datasets import Datasets, SpectrumDatasetOnOff
from gammapy.modeling.sampler import Sampler, SAMPLER_BACKENDS
from gammapy.modeling.models import (
    UniformPrior,
    LogUniformPrior,
    PowerLawSpectralModel,
    Models,
)

tested_backends = [_ for _ in SAMPLER_BACKENDS if importlib.util.find_spec(_)]


@pytest.fixture()
def datasets_sampler():
    datasets = Datasets()
    for obs_id in [23523, 23526]:
        dataset = SpectrumDatasetOnOff.read(
            f"$GAMMAPY_DATA/joint-crab/spectra/hess/pha_obs{obs_id}.fits"
        )
        datasets.append(dataset)
    return datasets


@pytest.fixture()
def models_sampler():
    pwl1 = PowerLawSpectralModel(index=2.3)
    pwl1.amplitude.prior = LogUniformPrior(min=1e-12, max=1e-10)
    pwl1.index.prior = UniformPrior(min=2, max=3)
    return Models([SkyModel(pwl1, name="source1")])


def test_invalid_backend_raises():
    with pytest.raises(ValueError, match="unknown_backend"):
        Sampler(backend="unknown_backend")


@pytest.mark.parametrize("backend", tested_backends)
@requires_data()
def test_run_missing_prior(backend, datasets_sampler, models_sampler):
    models_sampler[0].spectral_model.index.prior = None
    datasets_sampler.models = models_sampler

    sampler_opts = {"live_points": 300}
    sampler = Sampler(backend=backend, sampler_opts=sampler_opts)
    with pytest.raises(ValueError):
        sampler.run(datasets_sampler)


@pytest.mark.parametrize("backend", tested_backends)
@requires_data()
def test_run(backend, datasets_sampler, models_sampler):
    datasets_sampler.models = models_sampler

    sampler_opts = {"live_points": 300}
    sampler = Sampler(backend=backend, sampler_opts=sampler_opts)

    result = sampler.run(datasets_sampler)

    assert result.success
    assert sampler._sampler.min_num_live_points == sampler_opts["live_points"]
    assert (
        result.samples.shape[1]
        == datasets_sampler.models.parameters.free_unique_parameters.value.shape[0]
    )

    required_keys = [
        "logz",
        "logzerr",
        "posterior",
        "samples",
        "ncall",
        "insertion_order_MWW_test",
    ]
    assert set(required_keys).issubset(result.sampler_results.keys())

    assert (
        result.models.parameters["index"].value
        == result.sampler_results["posterior"]["mean"][0]
    )
    assert (
        result.models.parameters["amplitude"].value
        == result.sampler_results["posterior"]["mean"][1]
    )
    assert (
        result.models.parameters["index"].error
        == result.sampler_results["posterior"]["stdev"][0]
    )
    assert (
        result.models.parameters["amplitude"].error
        == result.sampler_results["posterior"]["stdev"][1]
    )

    assert_allclose(result.models.parameters["index"].value, 2.7, rtol=0.1)
    assert_allclose(result.models.parameters["amplitude"].value, 4e-11, rtol=0.1)
    assert_allclose(result.models.parameters["index"].error, 0.1, rtol=0.2)
    assert_allclose(result.models.parameters["amplitude"].error, 3.2e-12, rtol=0.2)

    assert result.models._covariance is None


@pytest.mark.parametrize("backend", tested_backends)
@requires_data()
def test_run_linked_params(backend, datasets_sampler, models_sampler):
    pwl1 = models_sampler[0].spectral_model
    pwl2 = PowerLawSpectralModel()
    pwl2.index = pwl1.index
    pwl2.amplitude = pwl1.amplitude

    models_sampler.append(SkyModel(pwl2, name="source2"))
    datasets_sampler.models = models_sampler

    sampler_opts = {"live_points": 300}
    sampler = Sampler(backend=backend, sampler_opts=sampler_opts)

    result = sampler.run(datasets_sampler)

    assert result.success
    assert sampler._sampler.min_num_live_points == sampler_opts["live_points"]
    assert (
        result.samples.shape[1]
        == datasets_sampler.models.parameters.free_unique_parameters.value.shape[0]
    )

    required_keys = [
        "logz",
        "logzerr",
        "posterior",
        "samples",
        "ncall",
        "insertion_order_MWW_test",
    ]
    assert set(required_keys).issubset(result.sampler_results.keys())

    assert (
        result.models.parameters["index"].value
        == result.sampler_results["posterior"]["mean"][0]
    )
    assert (
        result.models.parameters["amplitude"].value
        == result.sampler_results["posterior"]["mean"][1]
    )
    assert (
        result.models.parameters["index"].error
        == result.sampler_results["posterior"]["stdev"][0]
    )
    assert (
        result.models.parameters["amplitude"].error
        == result.sampler_results["posterior"]["stdev"][1]
    )

    assert_allclose(result.models.parameters["index"].value, 2.7, rtol=0.1)
    assert_allclose(result.models.parameters["amplitude"].value, 2e-11, rtol=0.1)
    assert_allclose(result.models.parameters["index"].error, 0.1, rtol=0.2)
    assert_allclose(result.models.parameters["amplitude"].error, 1.6e-12, rtol=0.2)

    assert result.models._covariance is None
