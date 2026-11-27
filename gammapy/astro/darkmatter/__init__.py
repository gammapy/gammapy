# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Dark matter spatial and spectral models."""

from gammapy.modeling.models import SPECTRAL_MODEL_REGISTRY

from .profiles import (
    BurkertProfile,
    DMProfile,
    EinastoProfile,
    IsothermalProfile,
    MooreProfile,
    NFWProfile,
    ZhaoProfile,
)
from .spectra import (
    ContinuumPrimaryFlux,
    DarkMatterSpectralModel,
    DarkMatterAnnihilationSpectralModel,
    DarkMatterDecaySpectralModel,
    HDMSpectraPrimaryFlux,
    PrimaryFlux,
)
from .utils import JFactory, add_factor_prior

__all__ = [
    "add_factor_prior",
    "ContinuumPrimaryFlux",
    "HDMSpectraPrimaryFlux",
    "PrimaryFlux",
    "DarkMatterSpectralModel",
    "DarkMatterAnnihilationSpectralModel",
    "DarkMatterDecaySpectralModel",
    "JFactory",
    "BurkertProfile",
    "DMProfile",
    "EinastoProfile",
    "IsothermalProfile",
    "MooreProfile",
    "NFWProfile",
    "ZhaoProfile",
]
SPECTRAL_MODEL_REGISTRY.append(DarkMatterSpectralModel)
SPECTRAL_MODEL_REGISTRY.append(DarkMatterAnnihilationSpectralModel)
SPECTRAL_MODEL_REGISTRY.append(DarkMatterDecaySpectralModel)
