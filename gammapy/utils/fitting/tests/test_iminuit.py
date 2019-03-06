# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
from .. import Parameter, Parameters, optimize_iminuit

pytest.importorskip("iminuit")


def fcn(parameters):
    x = parameters["x"].value
    y = parameters["y"].value
    z = parameters["z"].value
    x_opt, y_opt, z_opt = 2, 3e5, 4e-5
    x_err, y_err, z_err = 0.2, 3e4, 4e-6
    return ((x - x_opt) / x_err) ** 2 + ((y - y_opt) / y_err) ** 2 + ((z - z_opt) / z_err) ** 2


@pytest.fixture()
def pars():
    x = Parameter("x", 2.1)
    y = Parameter("y", 3.1, scale=1e5)
    z = Parameter("z", 4.1, scale=1e-5)
    return Parameters([x, y, z])


def test_iminuit_basic(pars):
    factors, info, minuit = optimize_iminuit(function=fcn, parameters=pars)

    assert info["success"]
    assert_allclose(fcn(pars), 0, atol=1e-5)

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


def test_iminuit_frozen(pars):
    pars["y"].frozen = True

    factors, info, minuit = optimize_iminuit(function=fcn, parameters=pars)

    assert info["success"]

    assert_allclose(pars["x"].value, 2, rtol=1e-4)
    assert_allclose(pars["y"].value, 3.1e5)
    assert_allclose(pars["z"].value, 4.e-5, rtol=1e-4)
    assert_allclose(fcn(pars), 0.111112, rtol=1e-5)


def test_iminuit_limits(pars):
    pars["y"].min = 301000

    factors, info, minuit = optimize_iminuit(function=fcn, parameters=pars)

    assert info["success"]

    # Check the result in parameters is OK
    assert_allclose(pars["x"].value, 2, rtol=1e-2)
    assert_allclose(pars["y"].value, 301000, rtol=1e-3)

    # Check that minuit sees the limit factors correctly
    states = minuit.get_param_states()
    assert not states[0]["has_limits"]

    y = states[1]
    assert y["has_limits"]
    assert_allclose(y["lower_limit"], 3.01)

    # The next assert can be added when we no longer test on iminuit 1.2
    # See https://github.com/gammapy/gammapy/pull/1771
    # assert states[1]["upper_limit"] is None
