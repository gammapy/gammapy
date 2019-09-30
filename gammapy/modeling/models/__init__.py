# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Built-in models in Gammapy."""
from .cube import *
from .spatial import *
from .spectral import *
from .spectral_cosmic_ray import *
from .spectral_crab import *
from .temporal import *


class ModelRegistry(list):
    """Gammapy model registry class."""
    def get_cls(self, tag):
        for cls in self:
            if hasattr(cls, "tag") and cls.tag == tag:
                return cls
        raise KeyError(f"No model found with tag: {tag!r}")


SPATIAL_MODELS = ModelRegistry(
    [
        ConstantSpatialModel,
        TemplateSpatialModel,
        DiskSpatialModel,
        GaussianSpatialModel,
        PointSpatialModel,
        ShellSpatialModel,
    ]
)
"""Built-in spatial models."""

SPECTRAL_MODELS = ModelRegistry(
    [
        ConstantSpectralModel,
        CompoundSpectralModel,
        PowerLawSpectralModel,
        PowerLaw2SpectralModel,
        ExpCutoffPowerLawSpectralModel,
        ExpCutoffPowerLaw3FGLSpectralModel,
        SuperExpCutoffPowerLaw3FGLSpectralModel,
        SuperExpCutoffPowerLaw4FGLSpectralModel,
        LogParabolaSpectralModel,
        TemplateSpectralModel,
        GaussianSpectralModel,
        LogGaussianSpectralModel,
        AbsorbedSpectralModel,
        NaimaSpectralModel,
        ScaleSpectralModel,
    ]
)
"""Built-in spectral models."""

TEMPORAL_MODELS = ModelRegistry(
    [
        ConstantTemporalModel,
        PhaseCurveTemplateTemporalModel,
        LightCurveTemplateTemporalModel,
    ]
)
"""Built-in temporal models."""

MODELS = ModelRegistry(
    SPATIAL_MODELS
    + SPECTRAL_MODELS
    + TEMPORAL_MODELS
    + [SkyModel, SkyDiffuseCube, BackgroundModel]
)
"""All built-in models."""


__all__ = [
    "SPATIAL_MODELS",
    "TEMPORAL_MODELS",
    "SPECTRAL_MODELS",
    "SkyModelBase",
    "SkyModels",
    "SkyModel",
    "SkyDiffuseCube",
    "BackgroundModel",
    "create_crab_spectral_model",
    "create_cosmic_ray_spectral_model",
    "Absorption",
    "SpatialModel",
    "SpectralModel",
    "TemporalModel",
]

__all__.extend(cls.__name__ for cls in SPATIAL_MODELS)
__all__.extend(cls.__name__ for cls in SPECTRAL_MODELS)
__all__.extend(cls.__name__ for cls in TEMPORAL_MODELS)
