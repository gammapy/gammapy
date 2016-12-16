# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import astropy.units as u
import logging
from ..utils.random import get_random_state
from .utils import calculate_predicted_counts
from .core import PHACountsSpectrum
from .observation import SpectrumObservation, SpectrumObservationList

__all__ = [
    'SpectrumSimulation'
]

log = logging.getLogger(__name__)


class SpectrumSimulation(object):
    """Simulate `~gammapy.spectrum.SpectrumObservation`.

    For a usage example see :gp-extra-notebook:`spectrum_simulation`

    Parameters
    ----------
    livetime : `~astropy.units.Quantity`
        Livetime
    source_model : `~gammapy.spectrum.models.SpectralModel`
        Source model
    aeff : `~gammapy.irf.EffectiveAreaTable`
        Effective Area
    edisp : `~gammapy.irf.EnergyDispersion`, optional
        Energy Dispersion
    e_reco : `~astropy.units.Quantity`, optional
        see :func:`gammapy.spectrum.utils.calculate_predicted_counts`
    background_model : `~gammapy.spectrum.models.SpectralModel`, optional
        Background model
    alpha : float, optional
        Exposure ratio between source and background
    """

    def __init__(self, livetime, source_model, aeff, edisp=None,
                 e_reco=None, background_model=None, alpha=None):
        self.livetime = livetime
        self.source_model = source_model
        self.aeff = aeff
        self.edisp = edisp
        self.e_reco = e_reco or edisp.e_reco.data
        self.background_model = background_model
        self.alpha = alpha

        self.on_vector = None
        self.off_vector = None
        self.obs = None
        self.result = SpectrumObservationList()

    @property
    def npred_source(self):
        """Predicted source `~gammapy.spectrum.CountsSpectrum`

        calls :func:`gammapy.spectrum.utils.calculate_predicted_counts`
        """
        npred = calculate_predicted_counts(livetime=self.livetime,
                                           aeff=self.aeff,
                                           edisp=self.edisp,
                                           model=self.source_model,
                                           e_reco=self.e_reco)
        return npred

    @property
    def npred_background(self):
        """Predicted background `~gammapy.spectrum.CountsSpectrum`

        calls :func:`gammapy.spectrum.utils.calculate_predicted_counts`
        """
        npred = calculate_predicted_counts(livetime=self.livetime,
                                           aeff=self.aeff,
                                           edisp=self.edisp,
                                           model=self.background_model,
                                           e_reco=self.e_reco)
        return npred

    def run(self, seed):
        """Simulate `~gammapy.spectrum.SpectrumObservationList`

        The seeds will be set as observation id. Previously produced results
        will be overwritten.

        Parameters
        ----------
        seed : array of ints
            Random number generator seeds
        """
        self.reset()
        n_obs = len(seed)
        log.info("Simulating {} observations".format(n_obs))
        for counter, current_seed in enumerate(seed):
            progress = ((counter + 1) / n_obs) * 100
            if progress % 10 == 0:
                log.info("Progress : {} %".format(progress))
            self.simulate_obs(seed=current_seed, obs_id=current_seed)
            self.result.append(self.obs)

    def reset(self):
        """Clear all results"""
        self.result = SpectrumObservationList()
        self.obs = None
        self.on_vector = None
        self.off_vector = None

    def simulate_obs(self, obs_id, seed='random-seed'):
        """Simulate one `~gammapy.spectrum.SpectrumObservation`.

        The result is stored as ``obs`` attribute

        Parameters
        ----------
        obs_id : int
            Observation identifier
        seed : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
            see :func:~`gammapy.utils.random.get_random_state`
        """
        random_state = get_random_state(seed)
        self.simulate_source_counts(random_state)
        if self.background_model is not None:
            self.simulate_background_counts(random_state)
        obs = SpectrumObservation(on_vector=self.on_vector,
                                  off_vector=self.off_vector,
                                  aeff=self.aeff,
                                  edisp=self.edisp)
        obs.obs_id = obs_id
        self.obs = obs

    def simulate_source_counts(self, rand):
        """Simulate source `~gammapy.spectrum.PHACountsSpectrum`

        Source counts are added to the on vector.

        Parameters
        ----------
        rand: `~numpy.random.RandomState`
            random state
        """
        on_counts = rand.poisson(self.npred_source.data.value)

        counts_kwargs = dict(energy=self.e_reco,
                             livetime=self.livetime,
                             creator=self.__class__.__name__)

        on_vector = PHACountsSpectrum(data=on_counts,
                                      backscal=1,
                                      **counts_kwargs)

        self.on_vector = on_vector

    def simulate_background_counts(self, rand):
        """Simulate background `~gammapy.spectrum.PHACountsSpectrum`

        Background counts are added to the on vector. Furthermore
        background counts divided by alpha are added to the off vector.

        TODO: At the moment the source IRFs are used. Make it possible to pass
        dedicated background IRFs.

        Parameters
        ----------
        rand: `~numpy.random.RandomState`
            random state
        """
        bkg_counts = rand.poisson(self.npred_background.data.value)
        off_counts = rand.poisson(self.npred_background.data.value / self.alpha)

        # Add background to on_vector
        self.on_vector.data += bkg_counts * u.ct

        # Create off vector
        off_vector = PHACountsSpectrum(energy=self.e_reco,
                                       data=off_counts,
                                       livetime=self.livetime,
                                       backscal=1. / self.alpha,
                                       is_bkg=True,
                                       creator=self.__class__.__name__)
        self.off_vector = off_vector
