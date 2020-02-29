from gammapy.utils.registry import Registry
from .core import *
from .flux_points import *
from .map import *
from .spectrum import *

DATASETS = Registry([
    MapDataset,
    SpectrumDatasetOnOff,
    FluxPointsDataset]
)

__all__ = [
    "Dataset",
    "Datasets",
    "MapDatasetOnOff",
    "SpectrumDataset"
]
__all__.extend(cls.__name__ for cls in DATASETS)
