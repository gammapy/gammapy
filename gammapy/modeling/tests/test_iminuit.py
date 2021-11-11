# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
from gammapy.modeling import Parameter
from gammapy.modeling.iminuit import confidence_iminuit, optimize_iminuit
from gammapy.modeling.models import ModelBase, Models

pytest.importorskip("iminuit")


class MyModel(ModelBase):
    x = Parameter("x", 2.1, error=0.2)
    y = Parameter("y", 3.1, scale=1e5, error=3e4)
    z = Parameter("z", 4.1, scale=1e-5, error=4e-6)
    name = "test"
    datasets_names = ["test"]


class MyDataset:
    def __init__(self, name="test"):
        self.name = name
        self.models = Models(MyModel())

    def fcn(self):
        x, y, z = [p.value for p in self.models.parameters]
        x_opt, y_opt, z_opt = 2, 3e5, 4e-5
        x_err, y_err, z_err = 0.2, 3e4, 4e-6
        return (
            ((x - x_opt) / x_err) ** 2
            + ((y - y_opt) / y_err) ** 2
            + ((z - z_opt) / z_err) ** 2
        )


def test_iminuit_basic():
    ds = MyDataset()
    pars = ds.models.parameters
    factors, info, minuit = optimize_iminuit(function=ds.fcn, parameters=pars)

    assert info["success"]
    assert_allclose(ds.fcn(), 0, atol=1e-5)

    # Check the result in parameters is OK
    assert_allclose(pars["x"].value, 2, rtol=1e-3)
    assert_allclose(pars["y"].value, 3e5, rtol=1e-3)
    # Precision of estimate on "z" is very poor (0.040488). Why is it so bad?
    assert_allclose(pars["z"].value, 4e-5, rtol=2e-2)

    # Check that minuit sees the parameter factors correctly
    assert_allclose(factors, [2, 3, 4], rtol=1e-3)
    assert_allclose(minuit.values["par_000_x"], 2, rtol=1e-3)
    assert_allclose(minuit.values["par_001_y"], 3, rtol=1e-3)
    assert_allclose(minuit.values["par_002_z"], 4, rtol=1e-3)


def test_iminuit_stepsize():
    ds = MyDataset()
    pars = ds.models.parameters
    factors, info, minuit = optimize_iminuit(function=ds.fcn, parameters=pars)

    assert info["success"]
    assert_allclose(ds.fcn(), 0, atol=1e-5)
    assert_allclose(pars["x"].value, 2, rtol=1e-3)
    assert_allclose(pars["y"].value, 3e5, rtol=1e-3)
    assert_allclose(pars["z"].value, 4e-5, rtol=2e-2)


def test_iminuit_frozen():
    ds = MyDataset()
    pars = ds.models.parameters
    pars["y"].frozen = True

    factors, info, minuit = optimize_iminuit(function=ds.fcn, parameters=pars)

    assert info["success"]

    assert_allclose(pars["x"].value, 2, rtol=1e-4)
    assert_allclose(pars["y"].value, 3.1e5)
    assert_allclose(pars["z"].value, 4.0e-5, rtol=1e-4)
    assert_allclose(ds.fcn(), 0.111112, rtol=1e-5)


def test_iminuit_limits():
    ds = MyDataset()
    pars = ds.models.parameters
    pars["y"].min = 301000

    factors, info, minuit = optimize_iminuit(function=ds.fcn, parameters=pars)

    assert info["success"]

    # Check the result in parameters is OK
    assert_allclose(pars["x"].value, 2, rtol=1e-2)
    assert_allclose(pars["y"].value, 301000, rtol=1e-3)

    # Check that minuit sees the limit factors correctly
    params = minuit.init_params

    assert not params["x"].has_limits
    assert params["par_001_y"].has_limits
    assert_allclose(params["par_001_y"].lower_limit, 3.01)
    assert params["par_001_y"].upper_limit is None


def test_opts():
    ds = MyDataset()
    pars = ds.models.parameters
    factors, info, minuit = optimize_iminuit(
        function=ds.fcn, parameters=pars, migrad_opts={"ncall": 20}, tol=1.0, strategy=2
    )
    assert info["nfev"] == 29
    assert minuit.tol == 1.0
    assert minuit.strategy == 2


def test_iminuit_confidence():
    ds = MyDataset()
    pars = ds.models.parameters
    factors, info, minuit = optimize_iminuit(function=ds.fcn, parameters=pars)

    assert_allclose(ds.fcn(), 0, atol=1e-5)

    par = pars["x"]
    par.min, par.max = 0, 10

    result = confidence_iminuit(
        function=ds.fcn, parameters=pars, parameter=par, sigma=1, reoptimize=True
    )

    assert result["success"]

    assert_allclose(result["errp"], 0.2)
    assert_allclose(result["errn"], 0.2)
