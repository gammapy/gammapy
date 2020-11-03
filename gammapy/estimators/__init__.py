# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Estimators."""
from gammapy.utils.registry import Registry
from .asmooth_map import *
from .core import *
from .excess_map import *
from .excess_profile import *
from .flux_point import *
from .lightcurve import *
from .profile import *
from .sensitivity import *
from .ts_map import *

ESTIMATOR_REGISTRY = Registry(
    [
        ExcessMapEstimator,
        TSMapEstimator,
        ASmoothMapEstimator,
        FluxPointsEstimator,
        LightCurveEstimator,
        SensitivityEstimator,
        ImageProfileEstimator,
        ExcessProfileEstimator,
    ]
)
"""Registry of estimator classes in Gammapy."""

__all__ = [
    "ESTIMATOR_REGISTRY",
    "FluxPoints",
    "LightCurve",
    "ImageProfile",
    "Estimator",
]
__all__.extend(cls.__name__ for cls in ESTIMATOR_REGISTRY)
