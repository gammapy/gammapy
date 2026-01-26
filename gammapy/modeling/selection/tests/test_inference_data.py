# Licensed under a 3-clause BSD style license - see LICENSE.rst
from gammapy.utils.testing import requires_data, requires_dependency
import pytest
import numpy as np
from gammapy.modeling.selection.inference_data import (
    inference_data_from_ultranest,
    generate_prior_samples,
    inference_data_from_sampler,
)

from numpy.testing import assert_allclose
from gammapy.modeling.models import SkyModel
from gammapy.datasets import Datasets, SpectrumDatasetOnOff
from gammapy.modeling.sampler import Sampler
from gammapy.modeling.models import (
    UniformPrior,
    LogUniformPrior,
    PowerLawSpectralModel,
    Models,
)


@pytest.fixture
def datasets():
    datasets = Datasets()
    dataset = SpectrumDatasetOnOff.read(
        "$GAMMAPY_DATA/joint-crab/spectra/hess/pha_obs23523.fits"
    )
    datasets.append(dataset)
    return datasets


@pytest.fixture
def ultranest_result(datasets):
    pwl1 = PowerLawSpectralModel(index=2.3)
    pwl1.amplitude.prior = LogUniformPrior(min=1e-12, max=1e-10)
    pwl1.index.prior = UniformPrior(min=2, max=3)

    models = Models([SkyModel(pwl1, name="source1")])
    datasets.models = models

    sampler_opts = {"live_points": 50}
    sampler = Sampler(backend="ultranest", sampler_opts=sampler_opts)

    ultranest_result = sampler.run(datasets)
    return ultranest_result


@requires_dependency("ultranest")
@requires_dependency("arviz")
@requires_data()
def test_inference_data_from_sampler_basic(ultranest_result, datasets):
    inference_data = inference_data_from_sampler(
        results=ultranest_result,
        datasets=datasets,
        backend="ultranest",
        n_prosterior_samples=None,
        n_prior_samples=None,
        predictives=False,
    )

    assert "posterior" in inference_data.groups()
    assert "sample_stats" not in inference_data.groups()
    assert "prior" not in inference_data.groups()

    n_samples = len(ultranest_result.sampler_results["samples"][:, 0])
    assert inference_data.posterior.sizes["draw"] == n_samples


@requires_dependency("ultranest")
@requires_dependency("arviz")
@requires_data()
def test_inference_data_from_sampler_with_options(ultranest_result, datasets):
    inference_data = inference_data_from_sampler(
        results=ultranest_result,
        datasets=datasets,
        backend="ultranest",
        n_prosterior_samples=10,
        n_prior_samples=5,
        predictives=True,
    )

    assert "sample_stats" in inference_data.groups()
    assert "posterior" in inference_data.groups()
    assert "prior" in inference_data.groups()
    assert "posterior_predictive" in inference_data.groups()
    assert "prior_predictive" in inference_data.groups()
    assert "log_likelihood" in inference_data.groups()
    assert "log_prior" in inference_data.groups()
    assert "observed_data" in inference_data.groups()

    assert inference_data.posterior.sizes["draw"] == 10
    assert inference_data.log_likelihood.sizes["draw"] == 10
    assert inference_data.log_prior.sizes["draw"] == 10

    assert inference_data.prior.sizes["draw"] == 5
    assert inference_data.prior_predictive.sizes["draw"] == 5

    n_pixel = datasets[0].mask.data.sum()
    assert inference_data.observed_data.sizes["pixel"] == n_pixel
    assert inference_data.log_likelihood.sizes["pixel"] == n_pixel
    assert inference_data.posterior_predictive.sizes["pixel"] == n_pixel
    assert inference_data.prior_predictive.sizes["pixel"] == n_pixel


@requires_dependency("arviz")
def test_inference_data_from_ultranest():
    import arviz

    result_dict = {
        "paramnames": ["x", "y"],
        "samples": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "weighted_samples": {
            "points": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "upoints": np.array([[0.1, 0.2], [0.3, 0.4]]),
            "weights": np.array([0.4, 0.6]),
            "logl": np.array([-1.0, -0.5]),
        },
    }

    # unweighted
    data = inference_data_from_ultranest(result_dict, weighted=False)
    assert isinstance(data, arviz.InferenceData)
    assert "posterior" in data.groups()

    posterior = data.posterior
    assert set(posterior.data_vars) == {"x", "y"}
    assert_allclose(posterior.x.data.flatten(), result_dict["samples"][:, 0])
    assert_allclose(posterior.y.data.flatten(), result_dict["samples"][:, 1])

    # weighted
    data = inference_data_from_ultranest(result_dict, weighted=True)
    assert isinstance(data, arviz.InferenceData)
    assert "posterior" in data.groups()
    assert "sample_stats" in data.groups()
    assert "weights" in data.sample_stats
    assert "log_likelihood" in data.sample_stats

    posterior = data.posterior
    assert set(posterior.data_vars) == {"x", "y"}
    assert_allclose(
        posterior.x.data.flatten(), result_dict["weighted_samples"]["points"][:, 0]
    )
    assert_allclose(
        posterior.y.data.flatten(), result_dict["weighted_samples"]["points"][:, 1]
    )


def test_generate_prior_samples():
    class DummyPrior:
        def _inverse_cdf(self, val):
            return val * 10

    class DummyParameter:
        def __init__(self):
            self.prior = DummyPrior()

    parameters = [DummyParameter(), DummyParameter()]
    samples = generate_prior_samples(parameters, n_prior_samples=3, random_seed=0)
    assert samples.shape == (3, 2)
    assert np.all(samples >= 0) and np.all(samples <= 10)
