# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Estimators."""

from gammapy.utils.registry import Registry

from .core import Estimator
from .energydependentmorphology import EnergyDependentMorphologyEstimator
from .map import ASmoothMapEstimator, ExcessMapEstimator, FluxMaps, TSMapEstimator
from .metadata import FluxMetaData
from .parameter import ParameterEstimator, ParameterSensitivityEstimator
from .points import (
    FluxCollectionEstimator,
    FluxPoints,
    FluxPointsEstimator,
    FluxProfileEstimator,
    LightCurveEstimator,
    RegularizedFluxPointsEstimator,
    SensitivityEstimator,
)
from .profile import ImageProfile, ImageProfileEstimator
from .resolvedestimator import ResolvedEstimator

__all__ = [
    "ASmoothMapEstimator",
    "Estimator",
    "ESTIMATOR_REGISTRY",
    "ExcessMapEstimator",
    "FluxCollectionEstimator",
    "FluxMaps",
    "FluxPoints",
    "FluxPointsEstimator",
    "FluxProfileEstimator",
    "ImageProfile",
    "ImageProfileEstimator",
    "LightCurveEstimator",
    "ParameterEstimator",
    "RegularizedFluxPointsEstimator",
    "SensitivityEstimator",
    "ParameterSensitivityEstimator",
    "TSMapEstimator",
    "EnergyDependentMorphologyEstimator",
    "FluxMetaData",
    "ResolvedEstimator",
]


ESTIMATOR_REGISTRY = Registry(
    [
        ExcessMapEstimator,
        TSMapEstimator,
        FluxCollectionEstimator,
        FluxPointsEstimator,
        ASmoothMapEstimator,
        LightCurveEstimator,
        RegularizedFluxPointsEstimator,
        SensitivityEstimator,
        ParameterSensitivityEstimator,
        FluxProfileEstimator,
        ParameterEstimator,
    ]
)
"""Registry of estimator classes in Gammapy."""
