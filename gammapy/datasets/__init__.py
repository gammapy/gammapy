from gammapy.utils.registry import Registry
from .core import *
from .flux_points import *
from .map import *
from .simulate import *
from .spectrum import *

DATASET_REGISTRY = Registry([MapDataset, SpectrumDatasetOnOff, FluxPointsDataset])
"""Registry of dataset classes in Gammapy."""

__all__ = [
    "DATASET_REGISTRY",
    "Dataset",
    "Datasets",
    "MapDatasetOnOff",
    "SpectrumDataset",
    "MapDatasetEventSampler",
]
__all__.extend(cls.__name__ for cls in DATASET_REGISTRY)
