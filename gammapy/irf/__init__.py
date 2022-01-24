# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Instrument response functions (IRFs).
"""
from gammapy.utils.registry import Registry
from .background import *
from .edisp import *
from .effective_area import *
from .io import *
from .psf import *
from .rad_max import *

IRF_REGISTRY = Registry(
    [
        EffectiveAreaTable2D,
        EnergyDispersion2D,
        PSF3D,
        EnergyDependentMultiGaussPSF,
        PSFKing,
        Background3D,
        Background2D,
        PSFMap,
        EDispKernelMap,
        RadMax2D,
        EDispMap,
    ]
)

__all__ = ["IRF_REGISTRY"]
__all__.extend(cls.__name__ for cls in IRF_REGISTRY)
