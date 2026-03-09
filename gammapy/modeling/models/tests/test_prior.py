# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from scipy.stats import norm
from gammapy.modeling import Parameter
from gammapy.modeling.models import (
    PRIOR_REGISTRY,
    GaussianPrior,
    GeneralizedGaussianPrior,
    Model,
    Models,
    SkyModel,
    UniformPrior,
    LogUniformPrior,
    SamplesKDEPrior,
)
from gammapy.utils.testing import assert_quantity_allclose

TEST_PRIORS = [
    dict(
        name="gaussian",
        model=GaussianPrior(mu=4.0, sigma=1.0),
        prior_0=0.0 * u.Unit(""),
        prior_1=1.0 * u.Unit(""),
        val_at_0=17.837877,
        val_at_1=10.837877,
        inverse_cdf_at_0=-np.inf,
        inverse_cdf_at_1=np.inf,
    ),
    dict(
        name="uniform",
        model=UniformPrior(min=0.0, max=10),
        prior_0=0.0 * u.Unit(""),
        prior_1=11.0 * u.Unit(""),
        val_at_0=4.60517,
        val_at_1=np.inf,
        val_with_weight_2=9.21034,
        inverse_cdf_at_0=0.0,
        inverse_cdf_at_1=10.0,
    ),
    dict(
        name="loguniform",
        model=LogUniformPrior(min=1e-14, max=1e-10),
        prior_0=1e-10 * u.Unit(""),
        prior_1=1e-14 * u.Unit(""),
        val_at_0=-41.61104824714522,
        val_at_1=-60.03172899109759,
        inverse_cdf_at_0=1e-14,
        inverse_cdf_at_1=1e-10,
    ),
    dict(
        name="gennorm",
        model=GeneralizedGaussianPrior(mu=4, sigma=1.0, eta=0.5),
        prior_0=0.0 * u.Unit(""),
        prior_1=1.0 * u.Unit(""),
        val_at_0=17.837877,
        val_at_1=10.837877,
        inverse_cdf_at_0=-np.inf,
        inverse_cdf_at_1=np.inf,
    ),
]


@pytest.mark.parametrize("prior", TEST_PRIORS)
def test_prior_evaluation(prior):
    model = prior["model"]
    # Test the evaluation of the prior at specific points
    assert_allclose(model(prior["prior_0"]), prior["val_at_0"], rtol=1e-7)
    assert_allclose(model(prior["prior_1"]), prior["val_at_1"], rtol=1e-7)

    # Test the inverse_cdf at specific points
    value_0 = model._inverse_cdf(0)
    value_1 = model._inverse_cdf(1)
    assert_allclose(value_0, prior["inverse_cdf_at_0"], rtol=1e-7)
    assert_allclose(value_1, prior["inverse_cdf_at_1"], rtol=1e-7)


@pytest.mark.parametrize("prior", TEST_PRIORS)
def test_prior_parameters(prior):
    model = prior["model"]
    # Check that all parameters of the model have type 'prior'
    for p in model.parameters:
        assert p.type == "prior"


def test_uniform_prior_weight():
    prior = TEST_PRIORS[1]
    model = prior["model"]
    # Test the uniform prior with a specific weight
    model.weight = 2.0
    value_0_weight = model(0.0 * u.Unit(""))
    assert_allclose(value_0_weight, prior["val_with_weight_2"], rtol=1e-7)


def test_to_from_dict():
    prior = TEST_PRIORS[1]
    model = prior["model"]
    model.weight = 2.0
    model_dict = model.to_dict()
    # Here we reverse the order of parameters list to ensure assignment is correct
    model_dict["prior"]["parameters"].reverse()

    model_class = PRIOR_REGISTRY.get_cls(model_dict["prior"]["type"])
    new_model = model_class.from_dict(model_dict)

    assert isinstance(new_model, UniformPrior)

    actual = [par.value for par in new_model.parameters]
    desired = [par.value for par in model.parameters]
    assert_quantity_allclose(actual, desired)
    assert_allclose(model.weight, new_model.weight, rtol=1e-7)

    new_model = Model.from_dict(model_dict)

    assert isinstance(new_model, UniformPrior)

    actual = [par.value for par in new_model.parameters]
    desired = [par.value for par in model.parameters]
    assert_quantity_allclose(actual, desired)
    assert_allclose(model.weight, new_model.weight, rtol=1e-7)


@pytest.mark.parametrize("prior", TEST_PRIORS)
def test_serialisation(prior, tmpdir):
    model = SkyModel.create(spectral_model="pl", name="crab")
    model.spectral_model.amplitude.prior = prior["model"]
    models = Models([model])
    filename = str(tmpdir / "model_prior.yaml")
    models.write(filename)

    loaded_models = Models.read(filename)
    loaded_model = loaded_models[0]
    loaded_prior = loaded_model.spectral_model.amplitude.prior

    assert isinstance(loaded_prior, type(prior["model"]))


def test_uniform_prior_auto_syncs_bounds_when_unset():
    # Test that UniformPrior automatically syncs parameter bounds when no explicit bounds are set
    p = Parameter("lon_0", value=0.5)
    assert np.isnan(p.min)
    assert np.isnan(p.max)
    p.prior = UniformPrior(min=0.0, max=1.0)
    assert_allclose(p.min, 0.0)
    assert_allclose(p.max, 1.0)


def test_loguniform_prior_auto_syncs_bounds_when_unset():
    # Test that LogUniformPrior automatically syncs parameter bounds when no explicit bounds are set
    p = Parameter("amplitude", value=1e-12)
    assert np.isnan(p.min)
    assert np.isnan(p.max)
    p.prior = LogUniformPrior(min=1e-14, max=1e-10)
    assert_allclose(p.min, 1e-14)
    assert_allclose(p.max, 1e-10)


def test_gaussian_prior_does_not_set_bounds():
    # Test that GaussianPrior doesn't set bounds (it never returns inf)
    p = Parameter("index", value=2.0)
    p.prior = GaussianPrior(mu=2.0, sigma=0.2)
    assert np.isnan(p.min)
    assert np.isnan(p.max)


def test_generalized_gaussian_prior_does_not_set_bounds():
    # Test that GeneralizedGaussianPrior doesn't set bounds (it never returns inf)
    p = Parameter("index", value=2.0)
    p.prior = GeneralizedGaussianPrior(mu=2.0, sigma=0.2)
    assert np.isnan(p.min)
    assert np.isnan(p.max)


def test_prior_modification_updates_bounds_dynamically():
    # Test that parameter bounds update automatically when prior bounds are modified
    p = Parameter("lon_0", value=0.5)
    p.prior = UniformPrior(min=-1.0, max=1.0)
    assert_allclose(p.min, -1.0)
    assert_allclose(p.max, 1.0)
    p.prior.min.value = -2.0
    p.prior.max.value = 2.0
    assert_allclose(p.min, -2.0)
    assert_allclose(p.max, 2.0)


def test_clearing_prior_restores_nan_bounds():
    # Test that clearing the prior restores nan bounds if they were synced
    p = Parameter("lon_0", value=0.5)
    p.prior = UniformPrior(min=-1.0, max=1.0)
    assert_allclose(p.min, -1.0)
    assert_allclose(p.max, 1.0)
    p.prior = None
    assert np.isnan(p.min)
    assert np.isnan(p.max)


def test_factor_min_max_use_synced_bounds():
    # Test that factor_min and factor_max correctly use the synced bounds
    p = Parameter("amplitude", value=1e-12, scale=1e-12)
    p.prior = UniformPrior(min=0.0, max=1e-10)
    assert_allclose(p.factor_min, 0.0)
    assert_allclose(p.factor_max, 1e-10 / 1e-12)


def test_samples_prior_empty_samples_raises():
    with pytest.raises(ValueError, match="at least one sample"):
        SamplesKDEPrior([])


def test_samples_prior_weights_length_mismatch_raises():
    samples = np.array([0.0, 1.0, 2.0])
    weights = np.array([1.0, 2.0])  # wrong length
    with pytest.raises(ValueError, match="same length"):
        SamplesKDEPrior(samples, weights=weights)


def test_samples_prior_inverse_cdf_monotonic_and_bounds():
    rng = np.random.default_rng(2)
    samples = rng.normal(loc=0.0, scale=1.0, size=500)
    prior = SamplesKDEPrior(samples)

    p = np.linspace(0.01, 0.99, 21)
    x_from_p = prior._inverse_cdf(p)

    # monotonic: higher p should give higher x
    diff = np.diff(x_from_p)
    assert np.all(diff >= -1e-6)

    # values should lie within sample range (with small tolerance)
    s_min, s_max = samples.min(), samples.max()
    assert x_from_p.min() >= s_min - 1e-6
    assert x_from_p.max() <= s_max + 1e-6

    # behaviour at "edges" via fill_value
    x_lo = prior._inverse_cdf(0.0)
    x_hi = prior._inverse_cdf(1.0)
    assert x_lo <= s_min + 1e-6
    assert x_hi >= s_max - 1e-6


def test_samples_prior_against_scipy_norm():
    # KDE + approximation + interpolation will not be exact;
    # keep tolerance loose but check basic consistency.

    rng = np.random.default_rng(12)
    mu, sigma = 1.5, 0.7
    samples = rng.normal(mu, sigma, size=2000)
    weights = rng.uniform(0.8, 1.2, size=samples.size)

    prior = SamplesKDEPrior(samples, weights=weights)

    assert "SamplesKDEPrior" in prior.tag
    assert prior._type == "prior"

    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 40)
    expected_val = -2 * norm(mu, sigma).logpdf(x)
    val = prior.evaluate(x)
    assert_allclose(val, expected_val, rtol=0.1)
    assert np.all(np.isfinite(val))

    p = np.linspace(0.1, 0.95, 11)
    x_prior = prior._inverse_cdf(p)
    x_ref = norm(mu, sigma).ppf(p)
    assert_allclose(x_prior, x_ref, rtol=0.1)
    assert np.all(np.isfinite(x_prior))


def test_samples_prior_serialization_roundtrip():
    rng = np.random.default_rng(42)
    samples = rng.normal(loc=1.0, scale=0.5, size=500)
    weights = rng.uniform(0.8, 1.2, size=samples.size)

    prior = SamplesKDEPrior(samples, weights=weights)

    data = prior.to_dict()
    prior2 = SamplesKDEPrior.from_dict(data)

    x = np.linspace(samples.min() - 1, samples.max() + 1, 30)
    assert_allclose(prior.evaluate(x), prior2.evaluate(x), rtol=0, atol=1e-10)

    p = np.linspace(0.05, 0.95, 9)
    assert_allclose(
        prior._inverse_cdf(p),
        prior2._inverse_cdf(p),
        rtol=0,
        atol=1e-10,
    )
