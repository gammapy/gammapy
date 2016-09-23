# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import astropy.units as u
from ..utils.random import get_random_state
from .utils import calculate_predicted_counts
from .core import PHACountsSpectrum
from .observation import SpectrumObservation

__all__ = [
    'SpectrumSimulation'
]


class SpectrumSimulation(object):
    """Simulate `~gammapy.spectrum.SpectrumObservation`.

    For an example how to use this class, see :ref:`spectrum-simulation`.

    Parameters
    ----------
    aeff : `~gammapy.irf.EffectiveAreaTable`
        Effective Area
    edisp : `~gammapy.irf.EnergyDispersion`,
        Energy Dispersion
    model : `~gammapy.spectrum.models.SpectralModel`
        Source model
    livetime : `~astropy.units.Quantity`
        Livetime
    """

    def __init__(self, aeff, edisp, model, livetime):
        self.aeff = aeff
        self.edisp = edisp
        self.model = model
        self.livetime = livetime

    @property
    def npred(self):
        """Predicted source `~gammapy.spectrum.CountsSpectrum`"""
        npred = calculate_predicted_counts(livetime=self.livetime,
                                           aeff=self.aeff,
                                           edisp=self.edisp,
                                           model=self.model)
        return npred

    def simulate_obs(self, seed='random-seed'):
        """Simulate `~gammapy.spectrum.SpectrumObservation`.

        Parameters
        ----------
        seed : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
            see :func:~`gammapy.utils.random.get_random_state`
        """
        rand = get_random_state(seed)
        on_counts = rand.poisson(self.npred.data)

        counts_kwargs = dict(energy=self.npred.energy,
                             livetime=self.livetime,
                             creator=self.__class__.__name__)

        on_vector = PHACountsSpectrum(data=on_counts,
                                      backscal=1,
                                      **counts_kwargs)

        obs = SpectrumObservation(on_vector=on_vector,
                                  aeff=self.aeff,
                                  edisp=self.edisp)
        return obs
