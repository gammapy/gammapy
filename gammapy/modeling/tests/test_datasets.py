# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
from gammapy.modeling import Datasets
from .test_fit import MyDataset


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


def test_names(datasets):
    assert datasets.names == ["test-1", "test-2"]
