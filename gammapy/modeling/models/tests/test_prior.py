# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from gammapy.modeling.models import (
    PRIOR_REGISTRY,
    GaussianPrior,
    Model,
    UniformPrior,
    LogUniformPrior,
)
from gammapy.utils.testing import assert_quantity_allclose

TEST_PRIORS = [
    dict(
        name="gaussian",
        model=GaussianPrior(mu=4.0, sigma=1.0),
        prior_0=0.0 * u.Unit(""),
        prior_1=1.0 * u.Unit(""),
        val_at_0=16.0,
        val_at_1=9.0,
        inverse_cdf_at_0=-np.inf,
        inverse_cdf_at_1=np.inf,
    ),
    dict(
        name="uniform",
        model=UniformPrior(min=0.0, max=10),
        prior_0=0.0 * u.Unit(""),
        prior_1=1.0 * u.Unit(""),
        val_at_0=1.0,
        val_at_1=0.0,
        val_with_weight_2=2.0,
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


@pytest.mark.parametrize("prior", TEST_PRIORS)
def test_uniform_prior_weight(prior):
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
