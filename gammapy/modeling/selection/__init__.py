# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Model selection."""

from .nested import select_nested_models, NestedModelSelection
from .bayesian import (
    BayesianModelSelection,
    BayesianModelSelectionResult,
    InferenceResult,
)

__all__ = [
    "BayesianModelSelection",
    "BayesianModelSelectionResult",
    "InferenceResult",
    "select_nested_models",
    "NestedModelSelection",
]
