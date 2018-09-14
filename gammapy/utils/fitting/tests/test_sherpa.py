# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
from numpy.testing import assert_allclose
from ...testing import requires_dependency
from .. import Parameter, Parameters, optimize_sherpa


def fcn(parameters):
    x = parameters["x"].value
    y = parameters["y"].value
    z = parameters["z"].value
    return (x - 2) ** 2 + (y - 3) ** 2 + (z - 4) ** 2


# TODO: levmar doesn't work yet; needs array of statval as return in likelihood
# optimiser='gridsearch' would require very low tolerance asserts, not added for now


@requires_dependency("sherpa")
@pytest.mark.parametrize("method", ["moncar", "simplex"])
def test_sherpa(method):
    pars = Parameters([Parameter("x", 2.2), Parameter("y", 3.4), Parameter("z", 4.5)])

    factors, info, _ = optimize_sherpa(function=fcn, parameters=pars, method=method)

    assert info["success"]
    assert info["nfev"] > 10
    assert_allclose(factors, [2, 3, 4], rtol=1e-2)
    assert_allclose(pars["x"].value, 2, rtol=1e-2)
    assert_allclose(pars["y"].value, 3, rtol=1e-2)
    assert_allclose(pars["z"].value, 4, rtol=1e-2)
