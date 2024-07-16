# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Gammapy high level interface (workflow)."""
from gammapy.utils.registry import Registry
from .config import WorkflowConfig
from .core import Workflow
from .steps import (
    DataSelectionWorkflowStep,
    DatasetsWorkflowStep,
    ExcessMapWorkflowStep,
    FitWorkflowStep,
    FluxPointsWorkflowStep,
    LightCurveWorkflowStep,
    ObservationsWorkflowStep,
)

__all__ = [
    "Workflow",
    "WorkflowConfig",
]

WORKFLOW_STEP_REGISTRY = Registry(
    [
        DataSelectionWorkflowStep,
        ObservationsWorkflowStep,
        DatasetsWorkflowStep,
        ExcessMapWorkflowStep,
        FitWorkflowStep,
        FluxPointsWorkflowStep,
        LightCurveWorkflowStep,
    ]
)
