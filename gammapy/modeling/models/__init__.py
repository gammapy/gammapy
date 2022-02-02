# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Built-in models in Gammapy."""
from gammapy.utils.registry import Registry
from .core import Model, Models, DatasetModels, ModelBase
from .cube import (
    SkyModel, FoVBackgroundModel, TemplateNPredModel,
    create_fermi_isotropic_diffuse_model,
)
from .spatial import (
    SpatialModel,
    PointSpatialModel,
    GaussianSpatialModel,
    GeneralizedGaussianSpatialModel,
    DiskSpatialModel,
    ShellSpatialModel,
    Shell2SpatialModel,
    ConstantSpatialModel,
    ConstantFluxSpatialModel,
    TemplateSpatialModel,
)
from .spectral import (
    scale_plot_flux,
    integrate_spectrum,
    SpectralModel,
    ConstantSpectralModel,
    CompoundSpectralModel,
    PowerLawSpectralModel,
    PowerLawNormSpectralModel,
    PowerLaw2SpectralModel,
    BrokenPowerLawSpectralModel,
    SmoothBrokenPowerLawSpectralModel,
    PiecewiseNormSpectralModel,
    ExpCutoffPowerLawSpectralModel,
    ExpCutoffPowerLawNormSpectralModel,
    ExpCutoffPowerLaw3FGLSpectralModel,
    SuperExpCutoffPowerLaw3FGLSpectralModel,
    SuperExpCutoffPowerLaw4FGLSpectralModel,
    LogParabolaSpectralModel,
    LogParabolaNormSpectralModel,
    TemplateSpectralModel,
    ScaleSpectralModel,
    EBLAbsorptionNormSpectralModel,
    NaimaSpectralModel,
    GaussianSpectralModel,
)

from .spectral_cosmic_ray import create_cosmic_ray_spectral_model
from .spectral_crab import (
    MeyerCrabSpectralModel,
    create_crab_spectral_model,
)
from .temporal import (
    TemporalModel,
    ConstantTemporalModel,
    LinearTemporalModel,
    ExpDecayTemporalModel,
    GaussianTemporalModel,
    LightCurveTemplateTemporalModel,
    PowerLawTemporalModel,
    SineTemporalModel,
)



__all__ = [
    "create_fermi_isotropic_diffuse_model",
    "DatasetModels",
    "FoVBackgroundModel",
    "Model",
    "MODEL_REGISTRY",
    "ModelBase",
    "Models",
    "SPATIAL_MODEL_REGISTRY",
    "SPECTRAL_MODEL_REGISTRY",
    "TemplateNPredModel",
    "TEMPORAL_MODEL_REGISTRY",
    "BrokenPowerLawSpectralModel",
    "CompoundSpectralModel",
    "ConstantFluxSpatialModel",
    "ConstantSpatialModel",
    "ConstantSpectralModel",
    "ConstantTemporalModel",
    "create_cosmic_ray_spectral_model",
    "create_crab_spectral_model",
    "DiskSpatialModel",
    "EBLAbsorptionNormSpectralModel",
    "ExpCutoffPowerLaw3FGLSpectralModel",
    "ExpCutoffPowerLawNormSpectralModel",
    "ExpCutoffPowerLawSpectralModel",
    "ExpDecayTemporalModel",
    "GaussianSpatialModel",
    "GaussianSpectralModel",
    "GaussianTemporalModel",
    "GeneralizedGaussianSpatialModel",
    "integrate_spectrum",
    "LightCurveTemplateTemporalModel",
    "LinearTemporalModel",
    "LogParabolaNormSpectralModel",
    "LogParabolaSpectralModel",
    "MeyerCrabSpectralModel",
    "NaimaSpectralModel",
    "PiecewiseNormSpectralModel",
    "PointSpatialModel",
    "PowerLaw2SpectralModel",
    "PowerLawNormSpectralModel",
    "PowerLawSpectralModel",
    "PowerLawTemporalModel",
    "scale_plot_flux",
    "ScaleSpectralModel",
    "Shell2SpatialModel",
    "ShellSpatialModel",
    "SineTemporalModel",
    "SkyModel",
    "SmoothBrokenPowerLawSpectralModel",
    "SpatialModel",
    "SpectralModel",
    "SuperExpCutoffPowerLaw3FGLSpectralModel",
    "SuperExpCutoffPowerLaw4FGLSpectralModel",
    "TemplateSpatialModel",
    "TemplateSpectralModel",
    "TemporalModel",
]


SPATIAL_MODEL_REGISTRY = Registry(
    [
        ConstantSpatialModel,
        TemplateSpatialModel,
        DiskSpatialModel,
        GaussianSpatialModel,
        GeneralizedGaussianSpatialModel,
        PointSpatialModel,
        ShellSpatialModel,
        Shell2SpatialModel,
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
        LinearTemporalModel,
        LightCurveTemplateTemporalModel,
        ExpDecayTemporalModel,
        GaussianTemporalModel,
        PowerLawTemporalModel,
        SineTemporalModel,
    ]
)
"""Registry of temporal models classes."""

MODEL_REGISTRY = Registry([SkyModel, FoVBackgroundModel, TemplateNPredModel])
"""Registry of model classes"""
