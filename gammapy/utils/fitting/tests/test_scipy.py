# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
from .. import Parameter, Parameters, optimize_scipy


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

@pytest.mark.parametrize("method", ["Nelder-Mead", "L-BFGS-B", "Powell", "BFGS"])
def test_scipy_basic(pars, method):
    factors, info, optimizer = optimize_scipy(function=fcn, parameters=pars, method=method)

    assert info["success"]
    assert_allclose(fcn(pars), 0, atol=1e-5)

    # Check the result in parameters is OK
    assert_allclose(pars["x"].value, 2, rtol=1e-3)
    assert_allclose(pars["y"].value, 3e5, rtol=1e-3)
    assert_allclose(pars["z"].value, 4e-5, rtol=2e-2)


@pytest.mark.parametrize("method", ["Nelder-Mead", "L-BFGS-B", "Powell"])
def test_scipy_frozen(pars, method):
    pars["y"].frozen = True

    factors, info, optimizer = optimize_scipy(function=fcn, parameters=pars, method=method)

    assert info["success"]

    assert_allclose(pars["x"].value, 2, rtol=1e-4)
    assert_allclose(pars["y"].value, 3.1e5)
    assert_allclose(pars["z"].value, 4.e-5, rtol=1e-4)
    assert_allclose(fcn(pars), 0.1111111, rtol=1e-5)


@pytest.mark.parametrize("method", ["L-BFGS-B"])
def test_scipy_limits(pars, method):
    pars["y"].min = 301000

    factors, info, minuit = optimize_scipy(function=fcn, parameters=pars, method=method)

    assert info["success"]

    # Check the result in parameters is OK
    assert_allclose(pars["x"].value, 2, rtol=1e-2)
    assert_allclose(pars["y"].value, 301000, rtol=1e-3)
