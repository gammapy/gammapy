# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
from gammapy.datasets import Datasets
from gammapy.modeling import Parameters, Parameter


class MyDataset:
    def __init__(self, name=""):
        self.name = name
        self.parameters = Parameters(
            [Parameter("x", 2), Parameter("y", 3e2), Parameter("z", 4e-2)]
        )
        self.data_shape = (1,)

    def stat_sum(self):
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


@pytest.fixture(scope="session")
def datasets():
    return Datasets([MyDataset(name="test-1"), MyDataset(name="test-2")])


def test_datasets_init(datasets):
    # Passing a Python list of `Dataset` objects should work
    Datasets(list(datasets))

    # Passing an existing `Datasets` object should work
    Datasets(datasets)


def test_datasets_types(datasets):
    assert datasets.is_all_same_type


def test_datasets_likelihood(datasets):
    likelihood = datasets.stat_sum()
    assert_allclose(likelihood, 0)


def test_datasets_str(datasets):
    assert "Datasets" in str(datasets)


def test_datasets_getitem(datasets):
    assert datasets["test-1"].name == "test-1"
    assert datasets["test-2"].name == "test-2"
