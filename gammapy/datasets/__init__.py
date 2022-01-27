from gammapy.utils.registry import Registry
from .core import Dataset, Datasets
from .flux_points import FluxPointsDataset
from .map import MapDataset, MapDatasetOnOff, create_map_dataset_geoms
from .simulate import MapDatasetEventSampler
from .spectrum import SpectrumDataset, SpectrumDatasetOnOff

DATASET_REGISTRY = Registry([MapDataset, SpectrumDatasetOnOff, FluxPointsDataset])
"""Registry of dataset classes in Gammapy."""

__all__ = [
    "DATASET_REGISTRY",
    # core
    "Dataset",
    "Datasets",
    # flux_points
    "FluxPointsDataset",
    # map
    "create_map_dataset_geoms",
    "MapDataset",
    "MapDatasetOnOff",
    # simulate
    "MapDatasetEventSampler",
    # spectrum
    "SpectrumDataset",
    "SpectrumDatasetOnOff",
]

__all__.extend(cls.__name__ for cls in DATASET_REGISTRY)
