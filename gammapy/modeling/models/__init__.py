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
    "ConstantModel": ConstantModel,
    "PowerLaw": PowerLaw,
    "PowerLaw2": PowerLaw2,
    "ExponentialCutoffPowerLaw": ExponentialCutoffPowerLaw,
    "ExponentialCutoffPowerLaw3FGL": ExponentialCutoffPowerLaw3FGL,
    "PLSuperExpCutoff3FGL": PLSuperExpCutoff3FGL,
    "PLSuperExpCutoff4FGL": PLSuperExpCutoff4FGL,
    "LogParabola": LogParabola,
    "TableModel": TableModel,
    "SpectralGaussian": SpectralGaussian,
    "SpectralLogGaussian": SpectralLogGaussian,
    "AbsorbedSpectralModel": AbsorbedSpectralModel,
    "Absorption": Absorption,
}
