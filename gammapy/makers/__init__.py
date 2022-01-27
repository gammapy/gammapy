from gammapy.utils.registry import Registry

from .core import Maker
from .background import (
    ReflectedRegionsFinder,
    ReflectedRegionsBackgroundMaker,
    AdaptiveRingBackgroundMaker,
    FoVBackgroundMaker,
    PhaseBackgroundMaker,
    RingBackgroundMaker,
)

from .map import MapDatasetMaker
from .reduce import DatasetsMaker
from .safe import SafeMaskMaker
from .spectrum import SpectrumDatasetMaker

MAKER_REGISTRY = Registry(
    [
        ReflectedRegionsBackgroundMaker,
        AdaptiveRingBackgroundMaker,
        FoVBackgroundMaker,
        PhaseBackgroundMaker,
        RingBackgroundMaker,
        SpectrumDatasetMaker,
        MapDatasetMaker,
        SafeMaskMaker,
        DatasetsMaker,
    ]
)
"""Registry of maker classes in Gammapy."""

__all__ = [
    "MAKER_REGISTRY",
    "Maker",
    "ReflectedRegionsFinder",
    "ReflectedRegionsBackgroundMaker",
    "AdaptiveRingBackgroundMaker",
    "FoVBackgroundMaker",
    "PhaseBackgroundMaker",
    "RingBackgroundMaker",
    "MapDatasetMaker",
    "DatasetsMaker",
    "SafeMaskMaker",
    "SpectrumDatasetMaker",
]

__all__.extend(cls.__name__ for cls in MAKER_REGISTRY)
