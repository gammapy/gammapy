# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Built-in models in Gammapy."""
from .cube import *
from .spatial import *
from .spectral import *
from .spectral_cosmic_ray import *
from .spectral_crab import *
from .temporal import *

SPATIAL_MODELS = {
    "TemplateSpatialModel": TemplateSpatialModel,
    "DiskSpatialModel": DiskSpatialModel,
    "GaussianSpatialModel": GaussianSpatialModel,
    "PointSpatialModel": PointSpatialModel,
    "ShellSpatialModel": ShellSpatialModel,
}

TIME_MODELS = {
    "PhaseCurveTemplateTemporalModel": PhaseCurveTemplateTemporalModel,
    "LightCurveTemplateTemporalModel": LightCurveTemplateTemporalModel,
}

# TODO: add support for these models writing their .from_dict()
# "NaimaModel":NaimaModel,
# "ScaleModel": ScaleModel,
SPECTRAL_MODELS = {
    "ConstantSpectralModel": ConstantSpectralModel,
    "PowerLawSpectralModel": PowerLawSpectralModel,
    "PowerLaw2SpectralModel": PowerLaw2SpectralModel,
    "ExpCutoffPowerLawSpectralModel": ExpCutoffPowerLawSpectralModel,
    "ExpCutoffPowerLaw3FGLSpectralModel": ExpCutoffPowerLaw3FGLSpectralModel,
    "SuperExpCutoffPowerLaw3FGLSpectralModel": SuperExpCutoffPowerLaw3FGLSpectralModel,
    "SuperExpCutoffPowerLaw4FGLSpectralModel": SuperExpCutoffPowerLaw4FGLSpectralModel,
    "LogParabolaSpectralModel": LogParabolaSpectralModel,
    "TemplateSpectralModel": TemplateSpectralModel,
    "SpectralGaussian": SpectralGaussian,
    "SpectralLogGaussian": SpectralLogGaussian,
    "AbsorbedSpectralModel": AbsorbedSpectralModel,
    "Absorption": Absorption,
}
