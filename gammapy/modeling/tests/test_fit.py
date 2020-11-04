# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Unit tests for the Fit class"""
import pytest
from numpy.testing import assert_allclose
from astropy.table import Table
from gammapy.datasets import Dataset
from gammapy.modeling import Fit, Parameter
from gammapy.modeling.models import Model, Models
from gammapy.utils.testing import requires_dependency

pytest.importorskip("iminuit")


class MyModel(Model):
    x = Parameter("x", 2)
    y = Parameter("y", 3e2)
    z = Parameter("z", 4e-2)
    name = "test"
    datasets_names = ["test"]
    type = "model"


class MyDataset(Dataset):
    tag = "MyDataset"

    def __init__(self, name="test"):
        self.name = name
        self._models = Models([MyModel(x=1.99, y=2.99e3, z=3.99e-2)])
        self.data_shape = (1,)
        self.meta_table = Table()

    @property
    def models(self):
        return self._models

    def stat_sum(self):
        # self._model.parameters = parameters
        x, y, z = [p.value for p in self.models.parameters]
        x_opt, y_opt, z_opt = 2, 3e2, 4e-2
        return (x - x_opt) ** 2 + (y - y_opt) ** 2 + (z - z_opt) ** 2

    def fcn(self):
        x, y, z = [p.value for p in self.models.parameters]
        x_opt, y_opt, z_opt = 2, 3e5, 4e-5
        x_err, y_err, z_err = 0.2, 3e4, 4e-6
        return (
            ((x - x_opt) / x_err) ** 2
            + ((y - y_opt) / y_err) ** 2
            + ((z - z_opt) / z_err) ** 2
        )

    def stat_array(self):
        """Statistic array, one value per data point."""


@pytest.mark.parametrize("backend", ["minuit"])
def test_run(backend):
    dataset = MyDataset()
    fit = Fit([dataset])
    result = fit.run(backend=backend)
    pars = result.parameters

    assert result.success is True

    assert_allclose(pars["x"].value, 2, rtol=1e-3)
    assert_allclose(pars["y"].value, 3e2, rtol=1e-3)
    assert_allclose(pars["z"].value, 4e-2, rtol=1e-3)

    assert_allclose(pars["x"].error, 1, rtol=1e-7)
    assert_allclose(pars["y"].error, 1, rtol=1e-7)
    assert_allclose(pars["z"].error, 1, rtol=1e-7)

    correlation = dataset.models.covariance.correlation
    assert_allclose(correlation[0, 1], 0, atol=1e-7)
    assert_allclose(correlation[0, 2], 0, atol=1e-7)
    assert_allclose(correlation[1, 2], 0, atol=1e-7)


@requires_dependency("sherpa")
@pytest.mark.parametrize("backend", ["minuit", "sherpa", "scipy"])
def test_optimize(backend):
    dataset = MyDataset()
    fit = Fit([dataset], store_trace=True)

    if backend == "scipy":
        kwargs = {"method": "L-BFGS-B"}
    else:
        kwargs = {}

    result = fit.optimize(backend=backend, **kwargs)
    pars = dataset.models.parameters

    assert result.success is True
    assert_allclose(result.total_stat, 0, atol=1)

    assert_allclose(pars["x"].value, 2, rtol=1e-3)
    assert_allclose(pars["y"].value, 3e2, rtol=1e-3)
    assert_allclose(pars["z"].value, 4e-2, rtol=1e-2)

    assert len(result.trace) == result.nfev


# TODO: add some extra covariance tests, in addition to run
# Probably mainly if error message is OK if optimize didn't run first.
# def test_covariance():


@pytest.mark.parametrize("backend", ["minuit"])
def test_confidence(backend):
    dataset = MyDataset()
    fit = Fit([dataset])
    fit.optimize(backend=backend)
    result = fit.confidence("x")

    assert result["success"] is True
    assert_allclose(result["errp"], 1)
    assert_allclose(result["errn"], 1)

    # Check that original value state wasn't changed
    assert_allclose(dataset.models.parameters["x"].value, 2)


@pytest.mark.parametrize("backend", ["minuit"])
def test_confidence_frozen(backend):
    dataset = MyDataset()
    dataset.models.parameters["x"].frozen = True
    fit = Fit([dataset])
    fit.optimize(backend=backend)
    result = fit.confidence("y")

    assert result["success"] is True
    assert_allclose(result["errp"], 1)
    assert_allclose(result["errn"], 1)


def test_stat_profile():
    dataset = MyDataset()
    fit = Fit([dataset])
    fit.run()
    result = fit.stat_profile("x", nvalues=3)

    assert_allclose(result["x_scan"], [0, 2, 4], atol=1e-7)
    assert_allclose(result["stat_scan"], [4, 0, 4], atol=1e-7)
    assert len(result["fit_results"]) == 0

    # Check that original value state wasn't changed
    assert_allclose(dataset.models.parameters["x"].value, 2)


def test_stat_profile_reoptimize():
    dataset = MyDataset()
    fit = Fit([dataset])
    fit.run()

    dataset.models.parameters["y"].value = 0
    result = fit.stat_profile("x", nvalues=3, reoptimize=True)

    assert_allclose(result["x_scan"], [0, 2, 4], atol=1e-7)
    assert_allclose(result["stat_scan"], [4, 0, 4], atol=1e-7)
    assert_allclose(
        result["fit_results"][0].total_stat, result["stat_scan"][0], atol=1e-7
    )


def test_stat_surface():
    dataset = MyDataset()
    fit = Fit([dataset])
    fit.run()
    x_values = [1, 2, 3]
    y_values = [2e2, 3e2, 4e2]
    result = fit.stat_surface("x", "y", x_values=x_values, y_values=y_values)

    assert_allclose(result["x_scan"], x_values, atol=1e-7)
    assert_allclose(result["y_scan"], y_values, atol=1e-7)
    expected_stat = [
        [1.0001e04, 1.0000e00, 1.0001e04],
        [1.0000e04, 0.0000e00, 1.0000e04],
        [1.0001e04, 1.0000e00, 1.0001e04],
    ]
    assert_allclose(list(result["stat_scan"]), expected_stat, atol=1e-7)
    assert len(result["fit_results"]) == 0

    # Check that original value state wasn't changed
    assert_allclose(dataset.models.parameters["x"].value, 2)
    assert_allclose(dataset.models.parameters["y"].value, 3e2)


def test_stat_surface_reoptimize():
    dataset = MyDataset()
    fit = Fit([dataset])
    fit.run()

    dataset.models.parameters["z"].value = 0
    x_values = [1, 2, 3]
    y_values = [2e2, 3e2, 4e2]
    result = fit.stat_surface(
        "x", "y", x_values=x_values, y_values=y_values, reoptimize=True
    )

    assert_allclose(result["x_scan"], x_values, atol=1e-7)
    assert_allclose(result["y_scan"], y_values, atol=1e-7)
    expected_stat = [
        [1.0001e04, 1.0000e00, 1.0001e04],
        [1.0000e04, 0.0000e00, 1.0000e04],
        [1.0001e04, 1.0000e00, 1.0001e04],
    ]

    assert_allclose(list(result["stat_scan"]), expected_stat, atol=1e-7)
    assert_allclose(
        result["fit_results"][0][0].total_stat, result["stat_scan"][0][0], atol=1e-7
    )


def test_minos_contour():
    dataset = MyDataset()
    dataset.models.parameters["x"].frozen = True
    fit = Fit([dataset])
    fit.optimize(backend="minuit")
    result = fit.minos_contour("y", "z")

    assert result["success"] is True

    x = result["y"]
    assert_allclose(len(x), 10)
    assert_allclose(x[0], 299, rtol=1e-5)
    assert_allclose(x[-1], 299.133975, rtol=1e-5)
    y = result["z"]
    assert_allclose(len(y), 10)
    assert_allclose(y[0], 0.04, rtol=1e-5)
    assert_allclose(y[-1], 0.54, rtol=1e-5)

    # Check that original value state wasn't changed
    assert_allclose(dataset.models.parameters["y"].value, 300)
