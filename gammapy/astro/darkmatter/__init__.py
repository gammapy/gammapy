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
    DarkMatterAnnihilationSpectralModel,
    DarkMatterDecaySpectralModel,
    PrimaryFlux,
)
from .utils import JFactory

__all__ = [
    "DarkMatterAnnihilationSpectralModel",
    "DarkMatterDecaySpectralModel",
    "JFactory",
    "PrimaryFlux",
    "BurkertProfile",
    "DMProfile",
    "EinastoProfile",
    "IsothermalProfile",
    "MooreProfile",
    "NFWProfile",
    "ZhaoProfile",
]

SPECTRAL_MODEL_REGISTRY.append(DarkMatterAnnihilationSpectralModel)
SPECTRAL_MODEL_REGISTRY.append(DarkMatterDecaySpectralModel)
