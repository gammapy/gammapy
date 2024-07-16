# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Gammapy high level interface (analysis)."""
from gammapy.utils.registry import Registry
from .config import AnalysisConfig
from .core import Analysis
from .steps import (
    DataSelectionAnalysisStep,
    DatasetsAnalysisStep,
    ExcessMapAnalysisStep,
    FitAnalysisStep,
    FluxPointsAnalysisStep,
    LightCurveAnalysisStep,
    ObservationsAnalysisStep,
)

__all__ = [
    "Analysis",
    "AnalysisConfig",
]

ANALYSIS_STEP_REGISTRY = Registry(
    [
        DataSelectionAnalysisStep,
        ObservationsAnalysisStep,
        DatasetsAnalysisStep,
        ExcessMapAnalysisStep,
        FitAnalysisStep,
        FluxPointsAnalysisStep,
        LightCurveAnalysisStep,
    ]
)
