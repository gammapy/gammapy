# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Spatial 2D image models."""
from .core import *

spatial_models = {
    "SkyDiffuseMap": SkyDiffuseMap,
    "SkyDisk": SkyDisk,
    "SkyEllipse": SkyEllipse,
    "SkyGaussian": SkyGaussian,
    "SkyGaussianElongated": SkyGaussianElongated,
    "SkyPointSource": SkyPointSource,
    "SkyShell": SkyShell,
}
