# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from astropy.table import Table
import astropy.units as u
from gammapy.modeling import Parameter, Parameters, PriorParameter, PriorParameters
from gammapy.modeling.models import GaussianPrior


@pytest.fixture
def default_parameter():
    return Parameter(
        "test",
        value=1e-10,
        min=None,
        max=None,
        frozen=False,
        unit="TeV",
        scale_method="scale10",
    )


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
    assert not par.frozen

    par = Parameter("spam", "42 deg")
    assert par.factor == 42
    assert par.scale == 1
    assert par.unit == "deg"

    with pytest.raises(TypeError):
        Parameter(1, 2)


def test_parameter_outside_limit(caplog):
    par = Parameter("spam", 50, min=0, max=40)
    par.check_limits()
    assert "WARNING" in [_.levelname for _ in caplog.records]
    message1 = "Value 50.0 is outside bounds [0.0, 40.0] for parameter 'spam'"
    assert message1 in [_.message for _ in caplog.records]


def test_parameter_scale():
    # Basic check how scale is used for value, min, max
    par = Parameter("spam", 420, "deg", 10, 400, 500)

    assert par.value == 420
    assert par.min == 400
    assert_allclose(par.scale, 10)
    assert_allclose(par.factor_min, 40)
    assert par.max == 500
    assert_allclose(par.factor_max, 50)

    par.value = 70
    assert par.scale == 10
    assert_allclose(par.factor, 7)


def test_parameter_quantity():
    par = Parameter("spam", 420, "deg", 10)

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
        ("factor1", 2e10, 1, 2e10),
        ("factor1", -2e10, 1, -2e10),
        # Check no scaling
        (None, 2e10, 2e10, 1),
    ],
)
def test_parameter_autoscale(method, value, factor, scale):
    par = Parameter("", value, scale_method=method)
    par.autoscale()
    assert_allclose(par.factor, factor)
    assert_allclose(par.scale, scale)
    assert isinstance(par.scale, float)


def test_parameter_scale_method_change():
    value = 2e10
    par = Parameter("", value, scale_method="scale10")
    par.autoscale()
    assert_allclose(par.factor, 2)
    assert_allclose(par.scale, 1e10)
    par.scale_method = "factor1"
    assert par.scale_method == "factor1"
    assert_allclose(par.factor, value)
    assert_allclose(par.scale, 1)
    par.autoscale()
    assert_allclose(par.factor, 1)
    assert_allclose(par.scale, value)


def test_parameter_scale_transform_change():
    value = 100
    par = Parameter("", value, scale_method=None, scale_transform="log")
    par.autoscale()
    assert_allclose(par.factor, np.log(value))
    assert_allclose(par.scale, 1)
    par.scale_transform = "sqrt"
    assert par.scale_transform == "sqrt"
    assert_allclose(par.factor, value)
    assert_allclose(par.scale, 1)
    par.autoscale()
    assert_allclose(par.factor, 10)
    assert_allclose(par.scale, 1)

    with pytest.raises(ValueError):
        par.scale_transform = "invalid"


@pytest.fixture()
def pars():
    return Parameters(
        [
            Parameter("spam", 42, "deg"),
            Parameter("ham", 99, "TeV", prior=GaussianPrior()),
        ]
    )


def test_parameters_basics(pars):
    # This applies a unit transformation
    pars["ham"].error = "10000 GeV"
    pars["spam"].error = 0.1
    assert_allclose(pars["spam"].error, 0.1)
    assert_allclose(pars[1].error, 10)


def test_parameters_copy(pars):
    pars2 = pars.copy()
    assert pars is not pars2
    assert pars[0] is not pars2[0]


def test_parameters_from_stack():
    a = Parameter("a", 1)
    b = Parameter("b", 2)
    c = Parameter("c", 3)

    pars = Parameters([a, b]) + Parameters([]) + Parameters([c])
    assert pars.names == ["a", "b", "c"]


def test_unique_parameters():
    a = Parameter("a", 1)
    b = Parameter("b", 2)
    c = Parameter("c", 3)
    parameters = Parameters([a, b, a, c])
    assert parameters.names == ["a", "b", "a", "c"]
    parameters_unique = parameters.unique_parameters
    assert parameters_unique.names == ["a", "b", "c"]


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

    with pytest.raises(ValueError):
        pars[Parameter("bam!", 99)]


def test_parameters_to_table(pars, tmp_path):
    pars["ham"].error = 1e-10
    pars["spam"]._link_label_io = "test"

    table = pars.to_table()
    assert len(table) == 2
    assert len(table.columns) == 10
    assert table["link"][0] == "test"
    assert table["link"][1] == ""

    assert table["prior"][0] == ""
    assert table["prior"][1] == "GaussianPrior"
    assert table["type"][1] == ""

    table.write(tmp_path / "test_parameters.fits")
    Table.read(tmp_path / "test_parameters.fits")


def test_parameters_create_table():
    table = Parameters._create_default_table()

    assert len(table) == 0
    assert len(table.columns) == 10

    assert table.colnames == [
        "type",
        "name",
        "value",
        "unit",
        "error",
        "min",
        "max",
        "frozen",
        "link",
        "prior",
    ]
    assert table.dtype == np.dtype(
        [
            ("type", "<U1"),
            ("name", "<U1"),
            ("value", "<f8"),
            ("unit", "<U1"),
            ("error", "<f8"),
            ("min", "<f8"),
            ("max", "<f8"),
            ("frozen", "?"),
            ("link", "<U1"),
            ("prior", "<U1"),
        ]
    )


def test_parameters_set_parameter_factors(pars):
    pars.set_parameter_factors([77, 78])
    assert_allclose(pars["spam"].factor, 77)
    assert_allclose(pars["spam"].scale, 1)
    assert_allclose(pars["ham"].factor, 78)
    assert_allclose(pars["ham"].scale, 1)


def test_parameters_s():
    pars = Parameters(
        [
            Parameter("", 20, scale_method="scale10"),
            Parameter("", 20, scale_method=None),
        ]
    )
    pars_dict = pars.to_dict()
    pars.autoscale()
    assert_allclose(pars[0].factor, 2)
    assert_allclose(pars[0].scale, 10)

    assert pars_dict[0]["scale_method"] == "scale10"
    assert pars_dict[1]["scale_method"] is None
    pars = Parameters.from_dict(pars_dict)
    pars.autoscale()
    assert_allclose(pars[0].factor, 2)
    assert_allclose(pars[0].scale, 10)
    assert pars[1].scale_method is None
    pars.autoscale()
    assert_allclose(pars[1].factor, 20)
    assert_allclose(pars[1].scale, 1)

    # test for backward compatibilty
    pars_dict[0]["is_norm"] = True
    pars = Parameters.from_dict(pars_dict)
    assert not hasattr(pars[0], "is_norm")


def test_parameter_scan_values():
    p = Parameter(name="test", value=0, error=1)

    values = p.scan_values

    assert len(values) == 11
    assert_allclose(values[[0, -1]], [-2, 2])
    assert_allclose(values[5], 0)

    p.scan_n_sigma = 3
    assert_allclose(p.scan_values[[0, -1]], [-3, 3])

    p.scan_min = -2
    p.scan_max = 3
    assert_allclose(p.scan_values[[0, -1]], [-2, 3])

    p.scan_n_values = 5
    assert len(p.scan_values) == 5

    p.interp = "log"
    p.scan_n_values = 3
    p.scan_min = 0.1
    p.scan_max = 10
    assert_allclose(p.scan_values, [0.1, 1, 10])


def test_update_from_dict(default_parameter):
    par = default_parameter
    par.autoscale()
    data = {
        "model": "gc",
        "type": "spectral",
        "name": "test2",
        "value": 3e-10,
        "min": 0,
        "max": np.nan,
        "frozen": True,
        "unit": "GeV",
        "prior": None,
    }
    par.update_from_dict(data)
    assert par.name == "test"
    assert_allclose(par.factor, 3)
    assert_allclose(par.value, 3e-10)
    assert par.unit == "GeV"
    assert_allclose(par.min, 0)
    assert par.max is np.nan
    assert par.frozen
    assert par.prior is None
    data = {
        "model": "gc",
        "type": "spectral",
        "name": "test2",
        "value": 3e-10,
        "min": 0,
        "max": np.nan,
        "frozen": "True",
        "unit": "GeV",
        "prior": None,
    }
    par.update_from_dict(data)
    assert par.frozen


def test_priorparameter_init():
    par = PriorParameter("spam", 11)
    assert par.name == "spam"
    assert par.factor == 11
    assert isinstance(par.factor, float)
    assert par.scale == 1
    assert isinstance(par.scale, float)
    assert par.value == 11
    assert isinstance(par.value, float)
    assert par.unit == ""
    assert par.min is np.nan
    assert par.max is np.nan


def test_priorparameter_repr():
    par = PriorParameter("spam", 42, "deg")
    assert repr(par).startswith("PriorParameter(name=")


def test_priorparameter_to_dict():
    par = Parameter("spam", 42, "deg")
    d = par.to_dict()
    assert isinstance(d["unit"], str)


@pytest.fixture()
def priorpars():
    return PriorParameters([PriorParameter("spam", 42), PriorParameter("ham", 99)])


def test_priorparameters_basics(priorpars):
    # This applies a unit transformation
    priorpars["ham"].error = "10000"
    priorpars["spam"].error = 0.1
    assert_allclose(priorpars["spam"].error, 0.1)
    assert_allclose(priorpars[1].error, 10000)


def test_priorparameters_to_table(priorpars):
    priorpars["ham"].vallue = 1e-10
    priorpars["spam"]._link_label_io = "test"
    table = priorpars.to_table()
    assert len(table) == 2
    assert len(table.columns) == 7
    assert table["name"][0] == "spam"
    assert table["value"][1] == 99


def test_parameter_set_min_max_error(default_parameter):
    par = default_parameter
    assert_equal(par.min, np.nan)
    assert_equal(par.max, np.nan)

    par.set_lim(min=1, max=10)
    assert par.min == 1
    assert par.max == 10

    par.set_lim(max=5)
    assert par.max == 5
    assert par.min == 1

    assert par.error == 0


def test_set_quantity_str_float(default_parameter):
    par = default_parameter
    assert par._set_quantity_str_float(1) == 1
    assert par._set_quantity_str_float("3 TeV") == 3
    assert par._set_quantity_str_float(2 * u.TeV) == 2
    assert par._set_quantity_str_float(4e3 * u.GeV) == 4
    with pytest.raises(u.UnitsError):
        par._set_quantity_str_float(1 * u.m)
