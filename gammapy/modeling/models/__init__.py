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


# this should probably not be done, but examples, notebooks and tutorials
# import Parameter from here
from ..parameter import Parameter, Parameters


__all__ = [
    "Parameter",
    "Parameters",
    # core
    "ModelBase",
    "Model",
    "Models",
    "DatasetModels",
    # cube
    "SkyModel",
    "FoVBackgroundModel",
    "TemplateNPredModel",
    "create_fermi_isotropic_diffuse_model",
    # spatial
    'SpatialModel',
    'PointSpatialModel',
    'GaussianSpatialModel',
    'GeneralizedGaussianSpatialModel',
    'DiskSpatialModel',
    'ShellSpatialModel',
    'Shell2SpatialModel',
    'ConstantSpatialModel',
    'ConstantFluxSpatialModel',
    'TemplateSpatialModel',
    # spectral
    'scale_plot_flux',
    'integrate_spectrum',
    'SpectralModel',
    'ConstantSpectralModel',
    'CompoundSpectralModel',
    'PowerLawSpectralModel',
    'PowerLawNormSpectralModel',
    'PowerLaw2SpectralModel',
    'BrokenPowerLawSpectralModel',
    'SmoothBrokenPowerLawSpectralModel',
    'PiecewiseNormSpectralModel',
    'ExpCutoffPowerLawSpectralModel',
    'ExpCutoffPowerLawNormSpectralModel',
    'ExpCutoffPowerLaw3FGLSpectralModel',
    'SuperExpCutoffPowerLaw3FGLSpectralModel',
    'SuperExpCutoffPowerLaw4FGLSpectralModel',
    'LogParabolaSpectralModel',
    'LogParabolaNormSpectralModel',
    'TemplateSpectralModel',
    'ScaleSpectralModel',
    'EBLAbsorptionNormSpectralModel',
    'NaimaSpectralModel',
    'GaussianSpectralModel',
    # spectral_cosmic_ray
    'create_cosmic_ray_spectral_model',
    # spectral_crab
    'MeyerCrabSpectralModel',
    'create_crab_spectral_model',
    # temporal
    'TemporalModel',
    'ConstantTemporalModel',
    'LinearTemporalModel',
    'ExpDecayTemporalModel',
    'GaussianTemporalModel',
    'LightCurveTemplateTemporalModel',
    'PowerLawTemporalModel',
    'SineTemporalModel',
    # here
    "MODEL_REGISTRY",
    "SPATIAL_MODEL_REGISTRY",
    "TEMPORAL_MODEL_REGISTRY",
    "SPECTRAL_MODEL_REGISTRY",
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


__all__.extend(cls.__name__ for cls in SPATIAL_MODEL_REGISTRY)
__all__.extend(cls.__name__ for cls in SPECTRAL_MODEL_REGISTRY)
__all__.extend(cls.__name__ for cls in TEMPORAL_MODEL_REGISTRY)
