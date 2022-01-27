# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Estimators."""
from gammapy.utils.registry import Registry
from .core import Estimator
from .map import (
    ASmoothMapEstimator,
    FluxMaps,
    ExcessMapEstimator,
    TSMapEstimator,
)
from .points import (
    FluxPoints,
    LightCurveEstimator,
    FluxProfileEstimator,
    FluxPointsEstimator,
    SensitivityEstimator,
)
from .profile import ImageProfile, ImageProfileEstimator


__all__ = [
    "ESTIMATOR_REGISTRY",
    "FluxPoints",
    "FluxMaps",
    "Estimator",
    "ASmoothMapEstimator",
    "FluxMaps",
    "ExcessMapEstimator",
    "TSMapEstimator",
    "LightCurveEstimator",
    "FluxProfileEstimator",
    "FluxPointsEstimator",
    "SensitivityEstimator",
    "ImageProfile",
    "ImageProfileEstimator",
]


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

