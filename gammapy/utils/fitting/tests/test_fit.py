# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Unit tests for the Fit class"""
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
from numpy.testing import assert_allclose
from ..parameter import Parameter, Parameters
from ..model import Model
from ..fit import Fit

pytest.importorskip("iminuit")


class MyModel(Model):
    """Dummy model class."""

    def __init__(self):
        self.parameters = Parameters(
            [Parameter("x", 2), Parameter("y", 3e2), Parameter("z", 4e-2)]
        )


class MyFit(Fit):
    def __init__(self):
        self._model = MyModel()

    def total_stat(self, parameters):
        # self._model.parameters = parameters
        x, y, z = [p.value for p in parameters]
        x_opt, y_opt, z_opt = 2, 3e2, 4e-2
        return (x - x_opt) ** 2 + (y - y_opt) ** 2 + (z - z_opt) ** 2


@pytest.mark.parametrize("backend", ["minuit"])
def test_run(backend):
    fit = MyFit()
    result = fit.run(
        optimize_opts={"backend": backend}, covariance_opts={"backend": backend}
    )
    pars = result.model.parameters

    assert_allclose(pars.error("x"), 1, rtol=1e-7)
    assert_allclose(pars.error("y"), 1, rtol=1e-7)
    assert_allclose(pars.error("z"), 1, rtol=1e-7)

    assert_allclose(pars.correlation[0, 1], 0, atol=1e-7)
    assert_allclose(pars.correlation[0, 2], 0, atol=1e-7)
    assert_allclose(pars.correlation[1, 2], 0, atol=1e-7)


@pytest.mark.parametrize("backend", ["minuit", "sherpa", "scipy"])
def test_optimize(backend):
    fit = MyFit()
    result = fit.optimize(backend=backend)
    pars = result.model.parameters

    assert result.success is True
    assert_allclose(result.total_stat, 0)

    assert_allclose(pars["x"].value, 2, rtol=1e-3)
    assert_allclose(pars["y"].value, 3e2, rtol=1e-3)
    assert_allclose(pars["z"].value, 4e-2, rtol=1e-3)

# TODO: add some extra covariance tests, in addition to run
# Probably mainly if error message is OK if optimize didn't run first.
# def test_covariance():


@pytest.mark.parametrize("backend", ["minuit"])
def test_confidence(backend):
    fit = MyFit()
    fit.optimize(backend=backend)
    result = fit.confidence("x")
    assert result["is_valid"] is True
    assert_allclose(result["lower"], -1)
    assert_allclose(result["upper"], +1)
