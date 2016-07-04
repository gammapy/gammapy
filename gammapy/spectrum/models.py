# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Spectral models for Gammapy.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = [
    'SpectralModel',
    'PowerLaw',
    'ExponentialCutoffPowerLaw',
]

# Note: Consider to move stuff from _models_old.py here

class SpectralModel(object):
    """Spectral model base class.
    """


class PowerLaw(SpectralModel):
    """Spectral power-law model.
    
    Parameters
    ----------
    index : float, `~astropy.units.Quantity`
    """
    def __init__(self, index, amplitude, reference):
        self.index = index
        self.amplitude = amplitude
        self.reference = reference


    def evaluate(self):
        pass

class ExponentialCutoffPowerLaw(SpectralModel):
    """Spectral exponential cutoff power-law model.
    """
