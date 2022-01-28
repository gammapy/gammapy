# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Dark matter spatial and spectral models."""
from .spectra import (
    PrimaryFlux,
    DarkMatterAnnihilationSpectralModel,
)
from .utils import JFactory


__all__ = [
    "DarkMatterAnnihilationSpectralModel",
    "JFactory",
    "PrimaryFlux",
]
