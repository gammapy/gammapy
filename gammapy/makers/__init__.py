from gammapy.utils.registry import Registry
from .background import *
from .core import *
from .map import *
from .safe import *
from .spectrum import *

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
    ]
)
"""Registry of maker classes in Gammapy."""

__all__ = ["MAKER_REGISTRY", "Maker"]
__all__.extend(cls.__name__ for cls in MAKER_REGISTRY)
