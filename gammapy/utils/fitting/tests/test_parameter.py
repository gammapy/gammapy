# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from .. import Parameter, Parameters


def test_parameter_init():
    par = Parameter("spam", 42, "deg")
    assert par.name == "spam"
    assert par.factor == 42
    assert isinstance(par.factor, float)
    assert par.scale == 1
    assert isinstance(par.scale, float)
    assert par.value == 42
    assert isinstance(par.value, float)
    assert par.unit == "deg"
    assert par.min is np.nan
    assert par.max is np.nan
    assert par.frozen is False

    par = Parameter("spam", "42 deg")
    assert par.factor == 42
    assert par.scale == 1
    assert par.unit == "deg"

    with pytest.raises(TypeError):
        Parameter(1, 2)


def test_parameter_scale():
    # Basic check how scale is used for value, min, max
    par = Parameter("spam", 42, "deg", 10, 400, 500)

    assert par.value == 420
    assert par.min == 400
    assert_allclose(par.factor_min, 40)
    assert par.max == 500
    assert_allclose(par.factor_max, 50)

    par.value = 70
    assert par.scale == 10
    assert_allclose(par.factor, 7)


def test_parameter_quantity():
    par = Parameter("spam", 42, "deg", 10)

    quantity = par.quantity
    assert quantity.unit == "deg"
    assert quantity.value == 420

    par.quantity = "70 deg"
    assert_allclose(par.factor, 7)
    assert par.scale == 10
    assert par.unit == "deg"


def test_parameter_repr():
    par = Parameter("spam", 42, "deg")
    assert repr(par).startswith("Parameter(name=")


def test_parameter_to_dict():
    par = Parameter("spam", 42, "deg")
    d = par.to_dict()
    assert isinstance(d["unit"], str)


@pytest.mark.parametrize(
    "method,value,factor,scale",
    [
        # Check method="scale10" in detail
        ("scale10", 2e-10, 2, 1e-10),
        ("scale10", 2e10, 2, 1e10),
        ("scale10", -2e-10, -2, 1e-10),
        ("scale10", -2e10, -2, 1e10),
        # Check that results are OK for very large numbers
        # Regression test for https://github.com/gammapy/gammapy/issues/1883
        ("scale10", 9e35, 9, 1e35),
        # Checks for the simpler method="factor1"
        ("factor1", 2e10, 2, 1e10),
        ("factor1", -2e10, -2, 1e10),
    ],
)
def test_parameter_autoscale(method, value, factor, scale):
    par = Parameter("", value)
    par.autoscale()
    assert_allclose(par.factor, factor)
    assert_allclose(par.scale, scale)
    assert isinstance(par.scale, float)


@pytest.fixture()
def pars():
    return Parameters([Parameter("spam", 42, "deg"), Parameter("ham", 99, "TeV")])


def test_parameters_basics(pars):
    # This applies a unit transformation
    pars.set_parameter_errors({"ham": "10000 GeV"})
    pars.set_error(0, 0.1)
    assert_allclose(pars.covariance, [[1e-2, 0], [0, 100]])
    assert_allclose(pars.correlation, [[1, 0], [0, 1]])
    assert_allclose(pars.error("spam"), 0.1)
    assert_allclose(pars.error(1), 10)


def test_parameters_getitem(pars):
    assert pars[1].name == "ham"
    assert pars["ham"].name == "ham"
    assert pars[pars[1]].name == "ham"

    with pytest.raises(TypeError):
        pars[42.3]

    with pytest.raises(IndexError):
        pars[3]

    with pytest.raises(IndexError):
        pars["lamb"]

    with pytest.raises(IndexError):
        pars[Parameter("bam!", 99)]


def test_parameters_to_table(pars):
    pars.set_error("ham", 1e-10 / 3)
    table = pars.to_table()
    assert len(table) == 2
    assert len(table.columns) == 7


def test_parameters_covariance_to_table(pars):
    with pytest.raises(ValueError):
        pars.covariance_to_table()

    pars.set_error("ham", 10)
    table = pars.covariance_to_table()
    assert_allclose(table["ham"][1], 100)


def test_parameters_set_parameter_factors(pars):
    pars.set_parameter_factors([77, 78])
    assert_allclose(pars["spam"].factor, 77)
    assert_allclose(pars["spam"].scale, 1)
    assert_allclose(pars["ham"].factor, 78)
    assert_allclose(pars["ham"].scale, 1)


def test_parameters_set_covariance_factors(pars):
    cov_factor = np.array([[3, 4], [7, 8]])
    pars.set_covariance_factors(cov_factor)
    assert_allclose(pars.covariance, cov_factor)


def test_parameters_autoscale():
    pars = Parameters([Parameter("", 20)])
    pars.autoscale()
    assert_allclose(pars[0].factor, 2)
    assert_allclose(pars[0].scale, 10)
