# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Instrument response functions (IRFs).
"""
from gammapy.utils.registry import Registry
from .background import Background3D, Background2D
from .edisp import (
    EnergyDispersion2D,
    EDispKernel,
    EDispKernelMap,
    EDispMap,
)
from .effective_area import EffectiveAreaTable2D
from .io import load_cta_irfs, load_irf_dict_from_file
from .psf import (
    PSFKernel,
    PSFMap,
    ParametricPSF,
    EnergyDependentMultiGaussPSF,
    PSFKing,
    PSF3D,
)
from .rad_max import RadMax2D


__all__ = [
    "Background3D",
    "Background2D",
    "EnergyDispersion2D",
    "EDispKernel",
    "EDispKernelMap",
    "EDispMap",
    "EffectiveAreaTable2D",
    "PSFKernel",
    "PSFMap",
    "ParametricPSF",
    "EnergyDependentMultiGaussPSF",
    "PSFKing",
    "PSF3D",
    "load_cta_irfs",
    "load_irf_dict_from_file",
    "IRF_REGISTRY",
]


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
