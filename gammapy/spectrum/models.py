# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Spectral models for Gammapy.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = [
    'SpectralModel',
    'PowerLaw',
    'ExponentialCutoffPowerLaw',
]


class SpectralModel(object):
    """Spectral model base class.
    """


class PowerLaw(SpectralModel):
    """Spectral power-law model.
    """


class ExponentialCutoffPowerLaw(SpectralModel):
    """Spectral exponential cutoff power-law model.
    """
