from gammapy.utils.registry import Registry
from .background import (
    AdaptiveRingBackgroundMaker,
    FoVBackgroundMaker,
    PhaseBackgroundMaker,
    ReflectedRegionsBackgroundMaker,
    ReflectedRegionsFinder,
    RegionsFinder,
    RingBackgroundMaker,
    WobbleRegionsFinder,
)
from .core import Maker
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
    "AdaptiveRingBackgroundMaker",
    "DatasetsMaker",
    "FoVBackgroundMaker",
    "Maker",
    "MAKER_REGISTRY",
    "MapDatasetMaker",
    "PhaseBackgroundMaker",
    "ReflectedRegionsBackgroundMaker",
    "ReflectedRegionsFinder",
    "RegionsFinder",
    "RingBackgroundMaker",
    "SafeMaskMaker",
    "SpectrumDatasetMaker",
    "WobbleRegionsFinder",
]
