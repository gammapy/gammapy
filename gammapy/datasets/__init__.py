# Licensed under a 3-clause BSD style license - see LICENSE.rst
from gammapy.utils.registry import Registry
from .core import Dataset, Datasets
from .flux_points import FluxPointsDataset
from .io import OGIPDatasetReader, OGIPDatasetWriter
from .map import (
    MapDataset,
    MapDatasetOnOff,
    create_map_dataset_from_observation,
    create_map_dataset_geoms,
)
from .metadata import MapDatasetMetaData
from .simulate import MapDatasetEventSampler, ObservationEventSampler
from .spectrum import SpectrumDataset, SpectrumDatasetOnOff
from .utils import apply_edisp, split_dataset

DATASET_REGISTRY = Registry(
    [
        MapDataset,
        MapDatasetOnOff,
        SpectrumDataset,
        SpectrumDatasetOnOff,
        FluxPointsDataset,
    ]
)

"""Registry of dataset classes in Gammapy."""

__all__ = [
    "create_map_dataset_from_observation",
    "create_map_dataset_geoms",
    "Dataset",
    "DATASET_REGISTRY",
    "Datasets",
    "FluxPointsDataset",
    "MapDataset",
    "MapDatasetEventSampler",
    "MapDatasetOnOff",
    "ObservationEventSampler",
    "OGIPDatasetWriter",
    "OGIPDatasetReader",
    "SpectrumDataset",
    "SpectrumDatasetOnOff",
    "MapDatasetMetaData",
    "apply_edisp",
    "split_dataset",
]
