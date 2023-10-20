# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
import astropy.units as u
from gammapy.modeling.models import PRIOR_REGISTRY, GaussianPrior, Model, UniformPrior
from gammapy.utils.testing import assert_quantity_allclose

TEST_PRIORS = [
    dict(
        name="gaussian",
        model=GaussianPrior(mu=4.0, sigma=1.0),
        val_at_0=16.0,
        val_at_1=9.0,
        val_with_weight_2=32.0,
    ),
    dict(
        name="uni",
        model=UniformPrior(min=0.0),
        val_at_0=0.0,
        val_at_1=1.0,
        val_with_weight_2=0.0,
    ),
]


@pytest.mark.parametrize("prior", TEST_PRIORS)
def test_priors(prior):
    model = prior["model"]
    for p in model.parameters:
        assert p.type == "prior"

    value_0 = model(0.0 * u.Unit(""))
    value_1 = model(1.0 * u.Unit(""))
    assert_allclose(value_0, prior["val_at_0"], rtol=1e-7)
    assert_allclose(value_1, prior["val_at_1"], rtol=1e-7)

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
