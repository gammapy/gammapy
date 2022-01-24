# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Estimators."""
from gammapy.utils.registry import Registry
from .core import *
from .map import *
from .points import *
from .profile import *

ESTIMATOR_REGISTRY = Registry(
    [
        ExcessMapEstimator,
        TSMapEstimator,
        FluxPointsEstimator,
        ASmoothMapEstimator,
        LightCurveEstimator,
        SensitivityEstimator,
        FluxProfileEstimator,
    ]
)
"""Registry of estimator classes in Gammapy."""

__all__ = [
    "ESTIMATOR_REGISTRY",
    "FluxPoints",
    "FluxMaps",
    "Estimator",
]
__all__.extend(cls.__name__ for cls in ESTIMATOR_REGISTRY)
