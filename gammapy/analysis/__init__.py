# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Gammapy high level interface (analysis)."""
from .config import AnalysisConfig
from .core import Analysis

__all__ = [
    "Analysis",
    "AnalysisConfig",
]
