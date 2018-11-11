# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Unit tests for the Fit class"""
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
from numpy.testing import assert_allclose
from ..parameter import Parameter, Parameters
from ..model import Model
from ..fit import Fit

pytest.importorskip("iminuit")


class MyData(object):
    """Dummy data class."""


class MyModel(Model):
    """Dummy model class."""

    def __init__(self):
        self.parameters = Parameters(
            [Parameter("x", 2), Parameter("y", 3e2), Parameter("z", 4e-2)]
        )


class MyFit(Fit):
    def __init__(self, model, data):
        self._model = model
        self.data = data

    def total_stat(self, parameters):
        # self._model.parameters = parameters
        x, y, z = [p.value for p in parameters]
        x_opt, y_opt, z_opt = 2, 3e2, 4e-2
        return (x - x_opt) ** 2 + (y - y_opt) ** 2 + (z - z_opt) ** 2


# TODO: parametrize the test, re-use it for the range of backends
# and optimiser / error estimator configurations we want to support
def test():
    data = MyData()
    model = MyModel()
    fit = MyFit(model, data)
    result = fit.run(optimize_opts={"backend": "minuit"})
    pars = result.model.parameters

    assert result.success is True
    assert_allclose(result.total_stat, 0)

    assert_allclose(pars["x"].value, 2, rtol=1e-3)
    assert_allclose(pars["y"].value, 3e2, rtol=1e-3)
    assert_allclose(pars["z"].value, 4e-2, rtol=1e-3)

    assert_allclose(pars.error("x"), 1, rtol=1e-7)
    assert_allclose(pars.error("y"), 1, rtol=1e-7)
    assert_allclose(pars.error("z"), 1, rtol=1e-7)

    # TODO: Add asserts once correlation method is added in the Parameters class.
    # assert_allclose(pars.correlation("b1", "b2"), -0.455547, rtol=1e-7)
    # assert_allclose(pars.correlation("b1", "b3"), -0.838906, rtol=1e-7)
    # assert_allclose(pars.correlation("b2", "b3"), 0.821333, rtol=1e-7)
