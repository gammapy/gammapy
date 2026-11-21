# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Tests for the actors module."""

import pytest
from gammapy.datasets import Datasets, MapDataset
from gammapy.datasets.actors import DatasetsActor
from gammapy.maps import MapAxis, WcsNDMap
from gammapy.utils.testing import requires_dependency


def get_simple_map_dataset(name="test"):
    """Create a simple MapDataset for testing."""
    axis = MapAxis.from_energy_bounds("1 TeV", "10 TeV", nbin=1)
    counts = WcsNDMap.create(npix=(10, 10), binsz=0.1, unit="", axes=[axis])
    background = WcsNDMap.create(npix=(10, 10), binsz=0.1, unit="", axes=[axis])
    return MapDataset(counts=counts, background=background, name=name)


@requires_dependency("ray")
def test_datasets_actor_init():
    """Test that DatasetsActor can be instantiated.

    This test covers the line `_ = self.name` in DatasetsActor.__init__
    which was added to trigger actors auto_init_wrapper.
    """
    dataset_1 = get_simple_map_dataset(name="test-1")
    dataset_2 = get_simple_map_dataset(name="test-2")
    datasets = Datasets([dataset_1, dataset_2])

    # This should trigger the line `_ = self.name` in __init__
    actors = DatasetsActor(datasets)

    assert len(actors) == 2
    assert actors.names == ["test-1", "test-2"]


@requires_dependency("ray")
def test_datasets_actor_name_property():
    """Test that accessing the name property works."""
    dataset = get_simple_map_dataset(name="test")
    datasets = Datasets([dataset])

    actors = DatasetsActor(datasets)

    # Access the name property
    names = actors.name
    assert names == ["test"]
