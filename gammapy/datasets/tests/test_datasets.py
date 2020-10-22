# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
from gammapy.datasets import Datasets
from gammapy.modeling.tests.test_fit import MyDataset


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
    assert_allclose(likelihood, 14472200.0002)


def test_datasets_str(datasets):
    assert "Datasets" in str(datasets)


def test_datasets_getitem(datasets):
    assert datasets["test-1"].name == "test-1"
    assert datasets["test-2"].name == "test-2"


def test_names(datasets):
    assert datasets.names == ["test-1", "test-2"]


def test_Datasets_mutation():
    dat = MyDataset(name="test-1")
    dats = Datasets([MyDataset(name="test-2"), MyDataset(name="test-3")])
    dats2 = Datasets([MyDataset(name="test-4"), MyDataset(name="test-5")])

    dats.insert(0, dat)
    assert dats.names == ["test-1", "test-2", "test-3"]

    dats.extend(dats2)
    assert dats.names == ["test-1", "test-2", "test-3", "test-4", "test-5"]

    dat3 = dats[3]
    dats.remove(dats[3])
    assert dats.names == ["test-1", "test-2", "test-3", "test-5"]
    dats.append(dat3)
    assert dats.names == ["test-1", "test-2", "test-3", "test-5", "test-4"]
    dats.pop(3)
    assert dats.names == ["test-1", "test-2", "test-3", "test-4"]

    with pytest.raises(ValueError, match="Dataset names must be unique"):
        dats.append(dat)
    with pytest.raises(ValueError, match="Dataset names must be unique"):
        dats.insert(0, dat)
    with pytest.raises(ValueError, match="Dataset names must be unique"):
        dats.extend(dats2)
