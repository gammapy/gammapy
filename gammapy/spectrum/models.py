# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Spectral models for Gammapy.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from ..extern.bunch import Bunch
from . import CountsSpectrum
import numpy as np


__all__ = [
    'SpectralModel',
    'PowerLaw',
    'ExponentialCutoffPowerLaw',
]

# Note: Consider to move stuff from _models_old.py here

class SpectralModel(object):
    """Spectral model base class.
    """
    def __call__(self, energy):
        kwargs = self.parameters
        kwargs.update(energy=energy)
        return self.evaluate(**kwargs)


class PowerLaw(SpectralModel):
    r"""Spectral power-law model.
    
    .. math:: 

        F(E) = F_0 \cdot \left( \frac{E}{E_0} \right)^{-\Gamma}

    Parameters
    ----------
    index : float, `~astropy.units.Quantity`
        :math:`\Gamma`
    amplitude: float, `~astropy.units.Quantity` 
        :math:`F_0`
    reference : float, `~astropy.units.Quantity` 
        :math:`E_0`
    """
    def __init__(self, index, amplitude, reference):
        self.parameters = Bunch(index = index,
                                amplitude = amplitude,
                                reference = reference)
        
    @staticmethod
    def evaluate(energy, index, amplitude, reference):
        return amplitude * ( energy / reference ) ** (-1 * index)

    def integral(self, emin, emax):
        """Integrate using analytic formula"""
        pars = self.parameters 
        
        val = -1 * pars.index + 1
        prefactor = pars.amplitude * pars.reference / val
        upper = (emax / pars.reference) ** val
        lower = (emin / pars.reference) ** val

        return prefactor * (upper - lower)


class ExponentialCutoffPowerLaw(SpectralModel):
    """Spectral exponential cutoff power-law model.
    """

