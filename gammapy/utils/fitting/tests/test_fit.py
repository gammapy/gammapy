# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Unit tests for the Fit class"""
import pytest
from numpy.testing import assert_allclose
from ..parameter import Parameter, Parameters
from ..fit import Fit
from ...testing import requires_dependency

pytest.importorskip("iminuit")


class MyDataset:
    def __init__(self):
        self.parameters = Parameters(
            [Parameter("x", 2), Parameter("y", 3e2), Parameter("z", 4e-2)]
        )
        self.data_shape = (1,)

    def likelihood(self):
        # self._model.parameters = parameters
        x, y, z = [p.value for p in self.parameters]
        x_opt, y_opt, z_opt = 2, 3e2, 4e-2
        return (x - x_opt) ** 2 + (y - y_opt) ** 2 + (z - z_opt) ** 2

    def fcn(self):
        x, y, z = [p.value for p in self.parameters]
        x_opt, y_opt, z_opt = 2, 3e5, 4e-5
        x_err, y_err, z_err = 0.2, 3e4, 4e-6
        return (
            ((x - x_opt) / x_err) ** 2
            + ((y - y_opt) / y_err) ** 2
            + ((z - z_opt) / z_err) ** 2
        )


@pytest.mark.parametrize("backend", ["minuit"])
def test_run(backend):
    dataset = MyDataset()
    fit = Fit(dataset)
    result = fit.run(
        optimize_opts={"backend": backend}, covariance_opts={"backend": backend}
    )
    pars = result.parameters

    assert result.success is True

    assert_allclose(pars["x"].value, 2, rtol=1e-3)
    assert_allclose(pars["y"].value, 3e2, rtol=1e-3)
    assert_allclose(pars["z"].value, 4e-2, rtol=1e-3)

    assert_allclose(pars.error("x"), 1, rtol=1e-7)
    assert_allclose(pars.error("y"), 1, rtol=1e-7)
    assert_allclose(pars.error("z"), 1, rtol=1e-7)

    assert_allclose(pars.correlation[0, 1], 0, atol=1e-7)
    assert_allclose(pars.correlation[0, 2], 0, atol=1e-7)
    assert_allclose(pars.correlation[1, 2], 0, atol=1e-7)


@requires_dependency("sherpa")
@pytest.mark.parametrize("backend", ["minuit", "sherpa", "scipy"])
def test_optimize(backend):
    dataset = MyDataset()
    fit = Fit(dataset)
    result = fit.optimize(backend=backend)
    pars = dataset.parameters

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
    dataset = MyDataset()
    fit = Fit(dataset)
    fit.optimize(backend=backend)
    result = fit.confidence("x")

    assert result["success"] is True
    assert_allclose(result["errp"], 1)
    assert_allclose(result["errn"], 1)

    # Check that original value state wasn't changed
    assert_allclose(dataset.parameters["x"].value, 2)


@pytest.mark.parametrize("backend", ["minuit"])
def test_confidence_frozen(backend):
    dataset = MyDataset()
    dataset.parameters["x"].frozen = True
    fit = Fit(dataset)
    fit.optimize(backend=backend)
    result = fit.confidence("y")

    assert result["success"] is True
    assert_allclose(result["errp"], 1)
    assert_allclose(result["errn"], 1)


def test_likelihood_profile():
    dataset = MyDataset()
    fit = Fit(dataset)
    fit.run()
    result = fit.likelihood_profile("x", nvalues=3)

    assert_allclose(result["values"], [0, 2, 4], atol=1e-7)
    assert_allclose(result["likelihood"], [4, 0, 4], atol=1e-7)

    # Check that original value state wasn't changed
    assert_allclose(dataset.parameters["x"].value, 2)


def test_likelihood_profile_reoptimize():
    dataset = MyDataset()
    fit = Fit(dataset)
    fit.run()

    dataset.parameters["y"].value = 0
    result = fit.likelihood_profile("x", nvalues=3, reoptimize=True)

    assert_allclose(result["values"], [0, 2, 4], atol=1e-7)
    assert_allclose(result["likelihood"], [4, 0, 4], atol=1e-7)


def test_minos_contour():
    dataset = MyDataset()
    dataset.parameters["x"].frozen = True
    fit = Fit(dataset)
    fit.optimize(backend="minuit")
    result = fit.minos_contour("y", "z")

    assert result["success"] is True

    x = result["x"]
    assert_allclose(len(x), 10)
    assert_allclose(x[0], 299, rtol=1e-5)
    assert_allclose(x[-1], 299.133975, rtol=1e-5)
    y = result["y"]
    assert_allclose(len(y), 10)
    assert_allclose(y[0], 0.04, rtol=1e-5)
    assert_allclose(y[-1], 0.54, rtol=1e-5)

    # Check that original value state wasn't changed
    assert_allclose(dataset.parameters["y"].value, 300)
