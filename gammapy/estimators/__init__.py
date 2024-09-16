# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Estimators."""
from gammapy.utils.registry import Registry
from .core import Estimator
from .energydependentmorphology import EnergyDependentMorphologyEstimator
from .map import ASmoothMapEstimator, ExcessMapEstimator, FluxMaps, TSMapEstimator
from .metadata import FluxMetaData
from .parameter import ParameterEstimator
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
        FluxProfileEstimator,
        ParameterEstimator,
    ]
)
"""Registry of estimator classes in Gammapy."""
