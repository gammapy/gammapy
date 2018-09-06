# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from ...modeling import Parameters, Parameter
from ...testing import requires_dependency
from .. import fit_iminuit


def fcn(parameters):
    x = parameters["x"].value
    y = parameters["y"].value
    z = parameters["z"].value
    return (x - 2) ** 2 + (y - 3) ** 2 + (z - 4) ** 2


@requires_dependency("iminuit")
def test_iminuit():
    pars = Parameters([Parameter("x", 2.1), Parameter("y", 3.1), Parameter("z", 4.1)])

    minuit = fit_iminuit(function=fcn, parameters=pars)

    assert_allclose(pars["x"].value, 2, rtol=1e-2)
    assert_allclose(pars["y"].value, 3, rtol=1e-2)
    assert_allclose(pars["z"].value, 4, rtol=1e-2)

    assert_allclose(minuit.values["par_000_x"], 2, rtol=1e-2)
    assert_allclose(minuit.values["par_001_y"], 3, rtol=1e-2)
    assert_allclose(minuit.values["par_002_z"], 4, rtol=1e-2)

    # Test freeze
    pars["x"].frozen = True
    minuit = fit_iminuit(function=fcn, parameters=pars)
    assert minuit.migrad_ok()
    assert minuit.list_of_fixed_param() == ["par_000_x"]

    # Test limits
    pars["y"].min = 4
    minuit = fit_iminuit(function=fcn, parameters=pars)
    assert minuit.migrad_ok()
    states = minuit.get_param_states()
    assert not states[0]["has_limits"]
    assert not states[2]["has_limits"]

    assert states[1]["has_limits"]
    assert states[1]["lower_limit"] == 4
    assert states[1]["upper_limit"] == 0
