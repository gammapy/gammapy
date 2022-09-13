from gammapy.utils.registry import Registry
from .core import Dataset, Datasets
from .flux_points import FluxPointsDataset
from .io import OGIPDatasetReader, OGIPDatasetWriter
from .map import MapDataset, MapDatasetOnOff, create_map_dataset_geoms
from .simulate import MapDatasetEventSampler
from .spectrum import SpectrumDataset, SpectrumDatasetOnOff

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
    "create_map_dataset_geoms",
    "Dataset",
    "DATASET_REGISTRY",
    "Datasets",
    "FluxPointsDataset",
    "MapDataset",
    "MapDatasetEventSampler",
    "MapDatasetOnOff",
    "OGIPDatasetWriter",
    "OGIPDatasetReader",
    "SpectrumDataset",
    "SpectrumDatasetOnOff",
]
