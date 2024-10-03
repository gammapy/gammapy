# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
from gammapy.modeling import Parameter
from gammapy.modeling.models import PRIOR_REGISTRY, GaussianPrior, Prior, UniformPrior
from gammapy.utils.testing import assert_quantity_allclose

TEST_PARAMETER = [dict(testparameter=Parameter(name="test", value=1))]


TEST_PRIORS = [
    dict(
        name="gaussian",
        model=GaussianPrior(mu=2.3, sigma=1.0, modelparameters=Parameter("index", 2.3)),
        val_at_default=0.0,
        val_at_0=5.29,
        val_with_weight_2=10.58,
    ),
    dict(
        name="uni",
        model=UniformPrior(min=0.0, modelparameters=Parameter("amplitude", 1e-12)),
        val_at_default=1.0,
        val_at_0=0.0,
        val_with_weight_2=0.0,
    ),
]


@pytest.mark.parametrize("prior", TEST_PRIORS)
def test_priors(prior):
    model = prior["model"]
    for p in model.parameters:
        assert p.type == "prior"

    value_default = model()
    model.modelparameters[0].value = 0.0
    value_0 = model()
    assert_allclose(value_default, prior["val_at_default"], rtol=1e-7)
    assert_allclose(value_0, prior["val_at_0"], rtol=1e-7)

    model.weight = 2.0
    value_2_weight = model()
    assert_allclose(value_2_weight, prior["val_with_weight_2"], rtol=1e-7)


def test_to_from_dict():
    prior = TEST_PRIORS[1]
    model = prior["model"]
    model.weight = 2.0
    model_dict = model.to_dict()
    # Here we reverse the order of parameters list to ensure assignment is correct
    model_dict["parameters"].reverse()

    model_class = PRIOR_REGISTRY.get_cls(model_dict["type"])
    new_model = model_class.from_dict(model_dict)

    assert isinstance(new_model, UniformPrior)

    actual = [par.value for par in new_model.parameters]
    desired = [par.value for par in model.parameters]
    assert_quantity_allclose(actual, desired)
    assert_allclose(model.weight, new_model.weight, rtol=1e-7)

    new_model = Prior.from_dict(model_dict)

    assert isinstance(new_model, UniformPrior)

    actual = [par.value for par in new_model.parameters]
    desired = [par.value for par in model.parameters]
    assert_quantity_allclose(actual, desired)
    assert_allclose(model.weight, new_model.weight, rtol=1e-7)
