# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Instrument response functions (IRFs)."""
from gammapy.utils.registry import Registry
from .background import Background2D, Background3D, BackgroundIRF
from .core import IRF, FoVAlignment, IRFMap
from .edisp import EDispKernel, EDispKernelMap, EDispMap, EnergyDispersion2D
from .effective_area import EffectiveAreaTable2D
from .io import load_irf_dict_from_file
from .psf import (
    PSF3D,
    EnergyDependentMultiGaussPSF,
    ParametricPSF,
    PSFKernel,
    PSFKing,
    PSFMap,
    RecoPSFMap,
)
from .rad_max import RadMax2D

__all__ = [
    "Background2D",
    "Background3D",
    "BackgroundIRF",
    "EDispKernel",
    "EDispKernelMap",
    "EDispMap",
    "EffectiveAreaTable2D",
    "EnergyDependentMultiGaussPSF",
    "EnergyDispersion2D",
    "FoVAlignment",
    "IRF_REGISTRY",
    "IRFMap",
    "IRF",
    "load_irf_dict_from_file",
    "ParametricPSF",
    "PSF3D",
    "PSFKernel",
    "PSFKing",
    "PSFMap",
    "RecoPSFMap",
    "RadMax2D",
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
        RecoPSFMap,
        EDispKernelMap,
        RadMax2D,
        EDispMap,
    ]
)
