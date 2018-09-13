# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing import assert_allclose
from ....extern import six
from .. import Parameter, Parameters


def test_parameter_init():
    par = Parameter("spam", 42, "deg")
    assert par.name == "spam"
    assert par.factor == 42
    assert par.scale == 1
    assert par.value == 42
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

    p = Parameter("spam", 42)
    with pytest.raises(TypeError):
        p.factor = "99"
    with pytest.raises(TypeError):
        p.scale = "99"


def test_parameter_value():
    par = Parameter("spam", 42, "deg", 10)

    value = par.value
    assert value == 420

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
    assert isinstance(d["unit"], six.string_types)


@pytest.fixture()
def pars():
    return Parameters([Parameter("spam", 42, "deg"), Parameter("ham", 99, "TeV")])


def test_parameters_basics(pars):
    # This applies a unit transformation
    pars.set_parameter_errors({"ham": "10000 GeV"})
    pars.set_error(0, 0.1)
    assert_allclose(pars.covariance, [[1e-2, 0], [0, 100]])
    assert_allclose(pars.error("spam"), 0.1)
    assert_allclose(pars.error(1), 10)


def test_parameters_to_table(pars):
    pars.set_error("ham", 1e-10 / 3)
    table = pars.to_table()
    assert len(table) == 2
    assert len(table.columns) == 6


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


def _test_parameters_set_covariance_factors(pars):
    cov_factor = [[3, 4], [7, 8]]
    pars.set_covariance_factors(cov_factor)

    assert isinstance(pars.covariance, np.ndarray)
    cov_value = [[0, 0], [0, 0]]
    assert_allclose(pars.covariance, cov_value)


def test_parameters_scale():
    pars = Parameters(
        [
            Parameter("", factor=10, scale=5),
            Parameter("", factor=10, scale=50),
            Parameter("", factor=100, scale=5),
            Parameter("", factor=-10, scale=1),
            Parameter("", factor=0, scale=1),
        ]
    )

    pars.autoscale()  # default: 'scale10'

    assert_allclose(pars[0].factor, 5)
    assert_allclose(pars[0].scale, 10)
    assert_allclose(pars[1].factor, 5)
    assert_allclose(pars[1].scale, 100)
    assert_allclose(pars[2].factor, 5)
    assert_allclose(pars[2].scale, 100)
    assert_allclose(pars[3].factor, -1)
    assert_allclose(pars[3].scale, 10)
    assert_allclose(pars[4].factor, 0)
    assert_allclose(pars[4].scale, 1)

    pars.autoscale("factor1")

    assert_allclose(pars[0].factor, 1)
    assert_allclose(pars[0].scale, 50)
    assert_allclose(pars[1].factor, 1)
    assert_allclose(pars[1].scale, 500)
    assert_allclose(pars[2].factor, 1)
    assert_allclose(pars[2].scale, 500)
    assert_allclose(pars[3].factor, 1)
    assert_allclose(pars[3].scale, -10)
    assert_allclose(pars[4].factor, 1)
    assert_allclose(pars[4].scale, 0)
