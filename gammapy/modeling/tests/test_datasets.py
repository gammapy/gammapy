# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
from gammapy.modeling import Datasets
from .test_fit import MyDataset


@pytest.fixture(scope="session")
def datasets():
    return Datasets([MyDataset(name="test-1"), MyDataset(name="test-2")])


class TestDatasets:
    @staticmethod
    def test_types(datasets):
        assert datasets.is_all_same_type

    @staticmethod
    def test_likelihood(datasets):
        likelihood = datasets.likelihood()
        assert_allclose(likelihood, 0)

    @staticmethod
    def test_str(datasets):
        assert "Datasets" in str(datasets)

    @staticmethod
    def test_getitem(datasets):
        assert datasets["test-1"].name == "test-1"
        assert datasets["test-2"].name == "test-2"
