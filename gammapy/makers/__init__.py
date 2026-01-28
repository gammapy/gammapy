# Licensed under a 3-clause BSD style license - see LICENSE.rst
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
from .events import EventDatasetMaker

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
        EventDatasetMaker,
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
    "EventDatasetMaker",
]
