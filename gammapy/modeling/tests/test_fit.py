# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Unit tests for the Fit class"""
import pytest
from numpy.testing import assert_allclose
from astropy.table import Table
from gammapy.datasets import Dataset
from gammapy.modeling import Fit, Parameter
from gammapy.modeling.models import ModelBase, Models
from gammapy.utils.testing import requires_dependency


class MyModel(ModelBase):
    x = Parameter("x", 2)
    y = Parameter("y", 3e2)
    z = Parameter("z", 4e-2)
    name = "test"
    datasets_names = ["test"]
    type = "model"


class MyDataset(Dataset):
    tag = "MyDataset"

    def __init__(self, name="test"):
        self._name = name
        self._models = Models([MyModel(x=1.99, y=2.99e3, z=3.99e-2)])
        self.data_shape = (1,)
        self.meta_table = Table()

    @property
    def models(self):
        return self._models

    def stat_sum(self):
        # self._model.parameters = parameters
        x, y, z = [p.value for p in self.models.parameters.unique_parameters]
        x_opt, y_opt, z_opt = 2, 3e2, 4e-2
        return (x - x_opt) ** 2 + (y - y_opt) ** 2 + (z - z_opt) ** 2

    def fcn(self):
        x, y, z = [p.value for p in self.models.parameters.unique_parameters]
        x_opt, y_opt, z_opt = 2, 3e5, 4e-5
        x_err, y_err, z_err = 0.2, 3e4, 4e-6
        return (
            ((x - x_opt) / x_err) ** 2
            + ((y - y_opt) / y_err) ** 2
            + ((z - z_opt) / z_err) ** 2
        )

    def stat_array(self):
        """Statistic array, one value per data point."""


@requires_dependency("sherpa")
@pytest.mark.parametrize("backend", ["sherpa", "scipy"])
def test_optimize_backend_and_covariance(backend):
    dataset = MyDataset()

    if backend == "scipy":
        kwargs = {"method": "L-BFGS-B"}
    else:
        kwargs = {}

    kwargs["backend"] = backend

    fit = Fit(optimize_opts=kwargs)
    result = fit.run([dataset])

    assert result is not None

    pars = dataset.models.parameters
    assert_allclose(pars["x"].value, 2, rtol=1e-3)
    assert_allclose(pars["y"].value, 3e2, rtol=1e-3)
    assert_allclose(pars["z"].value, 4e-2, rtol=1e-2)

    assert_allclose(pars["x"].error, 1, rtol=1e-7)
    assert_allclose(pars["y"].error, 1, rtol=1e-7)
    assert_allclose(pars["z"].error, 1, rtol=1e-7)

    correlation = dataset.models.covariance.correlation
    assert_allclose(correlation[0, 1], 0, atol=1e-7)
    assert_allclose(correlation[0, 2], 0, atol=1e-7)
    assert_allclose(correlation[1, 2], 0, atol=1e-7)


@pytest.mark.parametrize("backend", ["minuit"])
def test_run(backend):
    dataset = MyDataset()
    fit = Fit(backend=backend)
    result = fit.run([dataset])
    pars = dataset.models.parameters

    assert result.success
    assert result.optimize_result.method == "migrad"
    assert result.covariance_result.method == "hesse"
    assert result.covariance_result.success

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

    # Verify that the fit result models are independent of the dataset ones
    pars["x"].value = 3
    assert_allclose(result.parameters["x"].value, 2, rtol=1e-3)

    # check parameters from the result object
    pars = result.parameters
    assert_allclose(pars["x"].error, 1, rtol=1e-7)
    assert_allclose(pars["y"].error, 1, rtol=1e-7)
    assert_allclose(pars["z"].error, 1, rtol=1e-7)


def test_run_no_free_parameters():
    dataset = MyDataset()
    for par in dataset.models.parameters.free_parameters:
        par.frozen = True
    fit = Fit()
    with pytest.raises(ValueError, match="No free parameters for fitting"):
        fit.run(dataset)


@pytest.mark.parametrize("backend", ["minuit"])
def test_run_linked(backend):
    dataset = MyDataset()
    model1 = MyModel(x=1.99, y=2.99e3, z=3.99e-2)
    model1.name = "model1"
    model2 = MyModel(x=0.99, y=0.99e3, z=3.99e-2)
    model2.name = "model2"
    model2.x = model1.x
    model2.y = model1.y
    model2.z = model1.z

    dataset._models = Models([model1, model2])

    fit = Fit(backend=backend)
    fit.run([dataset])

    assert len(dataset.models.parameters.unique_parameters) == 3
    assert dataset.models.covariance.shape == (6, 6)
    expected = [
        [1.00000000e00, 1.69728073e-30, 4.76456033e-16],
        [1.69728073e-30, 1.00000000e00, 3.56230294e-15],
        [4.76456033e-16, 3.56230294e-15, 1.00000000e00],
    ]
    assert_allclose(dataset.models[0].covariance.data, expected)
    assert_allclose(dataset.models[1].covariance.data, expected)


@requires_dependency("sherpa")
@pytest.mark.parametrize("backend", ["minuit", "sherpa", "scipy"])
def test_optimize(backend):
    dataset = MyDataset()

    if backend == "scipy":
        kwargs = {"method": "L-BFGS-B"}
    else:
        kwargs = {}

    fit = Fit(store_trace=True, backend=backend, optimize_opts=kwargs)
    result = fit.optimize([dataset])
    pars = dataset.models.parameters

    assert result.success
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
    fit = Fit(backend=backend)
    fit.optimize([dataset])
    result = fit.confidence(datasets=[dataset], parameter="x")

    assert result["success"]
    assert_allclose(result["errp"], 1)
    assert_allclose(result["errn"], 1)

    # Check that original value state wasn't changed
    assert_allclose(dataset.models.parameters["x"].value, 2)


@pytest.mark.parametrize("backend", ["minuit"])
def test_confidence_frozen(backend):
    dataset = MyDataset()
    dataset.models.parameters["x"].frozen = True
    fit = Fit(backend=backend)
    fit.optimize([dataset])
    result = fit.confidence(datasets=[dataset], parameter="y")

    assert result["success"]
    assert_allclose(result["errp"], 1)
    assert_allclose(result["errn"], 1)


def test_stat_profile():
    dataset = MyDataset()
    fit = Fit()
    fit.run([dataset])
    dataset.models.parameters["x"].scan_n_values = 3
    result = fit.stat_profile(datasets=[dataset], parameter="x")

    assert_allclose(result["x_scan"], [0, 2, 4], atol=1e-7)
    assert_allclose(result["stat_scan"], [4, 0, 4], atol=1e-7)
    assert len(result["fit_results"]) == 0

    # Check that original value state wasn't changed
    assert_allclose(dataset.models.parameters["x"].value, 2)


def test_stat_profile_reoptimize():
    dataset = MyDataset()
    fit = Fit()
    fit.run([dataset])

    dataset.models.parameters["y"].value = 0
    dataset.models.parameters["x"].scan_n_values = 3
    result = fit.stat_profile(datasets=[dataset], parameter="x", reoptimize=True)

    assert_allclose(result["x_scan"], [0, 2, 4], atol=1e-7)
    assert_allclose(result["stat_scan"], [4, 0, 4], atol=1e-7)
    assert_allclose(
        result["fit_results"][0].total_stat, result["stat_scan"][0], atol=1e-7
    )


def test_stat_surface():
    dataset = MyDataset()
    fit = Fit()
    fit.run([dataset])

    x_values = [1, 2, 3]
    y_values = [2e2, 3e2, 4e2]

    dataset.models.parameters["x"].scan_values = x_values
    dataset.models.parameters["y"].scan_values = y_values
    result = fit.stat_surface(datasets=[dataset], x="x", y="y")

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
    fit = Fit()
    fit.run([dataset])

    x_values = [1, 2, 3]
    y_values = [2e2, 3e2, 4e2]

    dataset.models.parameters["z"].value = 0
    dataset.models.parameters["x"].scan_values = x_values
    dataset.models.parameters["y"].scan_values = y_values

    result = fit.stat_surface(datasets=[dataset], x="x", y="y", reoptimize=True)

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


def test_stat_contour():
    dataset = MyDataset()
    dataset.models.parameters["x"].frozen = True
    fit = Fit(backend="minuit")
    fit.optimize([dataset])
    result = fit.stat_contour(datasets=[dataset], x="y", y="z")

    assert result["success"]

    x = result["y"]
    assert len(x) in [10, 11]  # Behavior changed after iminuit>=2.13
    assert_allclose(x[0], 299, rtol=1e-5)
    assert_allclose(x[9], 299.133975, rtol=1e-5)
    y = result["z"]
    assert len(x) == len(y)
    assert len(y) in [10, 11]
    assert_allclose(y[0], 0.04, rtol=1e-5)
    assert_allclose(y[9], 0.54, rtol=1e-5)

    # Check that original value state wasn't changed
    assert_allclose(dataset.models.parameters["y"].value, 300)
