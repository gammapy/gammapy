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
    JointSensitivityEstimator,
)
from .profile import ImageProfile, ImageProfileEstimator

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
    "JointSensitivityEstimator",
    "ParameterSensitivityEstimator",
    "TSMapEstimator",
    "EnergyDependentMorphologyEstimator",
    "FluxMetaData",
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
