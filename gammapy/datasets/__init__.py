from gammapy.utils.registry import Registry
from .core import *
from .flux_points import *
from .map import *
from .simulate import *
from .spectrum import *

DATASETS = Registry(
    [
        MapDataset,
        SpectrumDatasetOnOff,
        FluxPointsDataset
    ]
)
"""Registry of dataset classes in Gammapy."""

__all__ = [
    "DATASETS",
    "Dataset",
    "Datasets",
    "MapDatasetOnOff",
    "SpectrumDataset"
]
__all__.extend(cls.__name__ for cls in DATASETS)
