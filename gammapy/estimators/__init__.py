# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Estimators."""

from gammapy.utils.registry import Registry
from .core import Estimator
from .energydependentmorphology import EnergyDependentMorphologyEstimator
from .map import ASmoothMapEstimator, ExcessMapEstimator, FluxMaps, TSMapEstimator
from .metadata import FluxMetaData
from .parameter import ParameterEstimator, ParameterSensitivityEstimator
from .points import (
    FluxPoints,
    FluxPointsEstimator,
    FluxProfileEstimator,
    LightCurveEstimator,
    SensitivityEstimator,
)
from .profile import ImageProfile, ImageProfileEstimator

__all__ = [
    "ASmoothMapEstimator",
    "Estimator",
    "ESTIMATOR_REGISTRY",
    "ExcessMapEstimator",
    "FluxMaps",
    "FluxPoints",
    "FluxPointsEstimator",
    "FluxProfileEstimator",
    "ImageProfile",
    "ImageProfileEstimator",
    "LightCurveEstimator",
    "ParameterEstimator",
    "SensitivityEstimator",
    "ParameterSensitivityEstimator",
    "TSMapEstimator",
    "EnergyDependentMorphologyEstimator",
    "FluxMetaData",
]


ESTIMATOR_REGISTRY = Registry(
    [
        ExcessMapEstimator,
        TSMapEstimator,
        FluxPointsEstimator,
        ASmoothMapEstimator,
        LightCurveEstimator,
        SensitivityEstimator,
        ParameterSensitivityEstimator,
        FluxProfileEstimator,
        ParameterEstimator,
    ]
)
"""Registry of estimator classes in Gammapy."""
