# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Built-in models in Gammapy."""
from gammapy.utils.registry import Registry
from .core import *
from .cube import *
from .spatial import *
from .spectral import *
from .spectral_cosmic_ray import *
from .spectral_crab import *
from .temporal import *

SPATIAL_MODEL_REGISTRY = Registry(
    [
        ConstantSpatialModel,
        TemplateSpatialModel,
        DiskSpatialModel,
        GaussianSpatialModel,
        GeneralizedGaussianSpatialModel,
        PointSpatialModel,
        ShellSpatialModel,
    ]
)
"""Registry of spatial model classes."""

SPECTRAL_MODEL_REGISTRY = Registry(
    [
        ConstantSpectralModel,
        CompoundSpectralModel,
        PowerLawSpectralModel,
        PowerLaw2SpectralModel,
        BrokenPowerLawSpectralModel,
        SmoothBrokenPowerLawSpectralModel,
        PiecewiseNormSpectralModel,
        ExpCutoffPowerLawSpectralModel,
        ExpCutoffPowerLaw3FGLSpectralModel,
        SuperExpCutoffPowerLaw3FGLSpectralModel,
        SuperExpCutoffPowerLaw4FGLSpectralModel,
        LogParabolaSpectralModel,
        TemplateSpectralModel,
        GaussianSpectralModel,
        EBLAbsorptionNormSpectralModel,
        NaimaSpectralModel,
        ScaleSpectralModel,
        PowerLawNormSpectralModel,
        LogParabolaNormSpectralModel,
        ExpCutoffPowerLawNormSpectralModel,
    ]
)
"""Registry of spectral model classes."""

TEMPORAL_MODEL_REGISTRY = Registry(
    [
        ConstantTemporalModel,
        LightCurveTemplateTemporalModel,
        ExpDecayTemporalModel,
        GaussianTemporalModel,
    ]
)
"""Registry of temporal models classes."""

MODEL_REGISTRY = Registry([SkyModel, FoVBackgroundModel, BackgroundModel])
"""Registry of model classes"""


__all__ = [
    "MODEL_REGISTRY",
    "SPATIAL_MODEL_REGISTRY",
    "TEMPORAL_MODEL_REGISTRY",
    "SPECTRAL_MODEL_REGISTRY",
    "Models",
    "SkyModel",
    "create_crab_spectral_model",
    "create_cosmic_ray_spectral_model",
    "SpatialModel",
    "SpectralModel",
    "TemporalModel",
]

__all__.extend(cls.__name__ for cls in SPATIAL_MODEL_REGISTRY)
__all__.extend(cls.__name__ for cls in SPECTRAL_MODEL_REGISTRY)
__all__.extend(cls.__name__ for cls in TEMPORAL_MODEL_REGISTRY)
