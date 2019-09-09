# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Built-in models in Gammapy."""
from .cube import *
from .spatial import *
from .spectrum import *
from .time import *

SPATIAL_MODELS = {
    "SkyDiffuseMap": SkyDiffuseMap,
    "SkyDisk": SkyDisk,
    "SkyEllipse": SkyEllipse,
    "SkyGaussian": SkyGaussian,
    "SkyGaussianElongated": SkyGaussianElongated,
    "SkyPointSource": SkyPointSource,
    "SkyShell": SkyShell,
}
