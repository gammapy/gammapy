from gammapy.utils.registry import Registry
from .background import *
from .core import *
from .map import *
from .safe import *
from .spectrum import *

MAKERS = Registry([
    ReflectedRegionsBackgroundMaker,
    AdaptiveRingBackgroundMaker,
    FoVBackgroundMaker,
    PhaseBackgroundMaker,
    RingBackgroundMaker,
    SpectrumDatasetMaker,
    MapDatasetMaker,
    SafeMaskMaker,
])

__all__ = ["Maker"]
__all__.extend(cls.__name__ for cls in MAKERS)

