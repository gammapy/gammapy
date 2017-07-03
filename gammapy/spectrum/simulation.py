# Licensed under a 3 - clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from collections import OrderedDict
import logging
import astropy.units as u
from ..utils.random import get_random_state
from ..utils.energy import EnergyBounds
from .utils import CountsPredictor
from .core import PHACountsSpectrum
from . import models
from .observation import SpectrumObservation, SpectrumObservationList

__all__ = [
    'SpectrumSimulation',
    'SpectrumEventSampler'
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
    background_model : `~gammapy.spectrum.models.SpectralModel`, optional
        Background model
    alpha : float, optional
        Exposure ratio between source and background
    """

    def __init__(self, livetime, source_model, aeff, edisp=None,
                 background_model=None, alpha=None):
        self.livetime = livetime
        self.source_model = source_model
        self.aeff = aeff
        self.edisp = edisp
        self.background_model = background_model
        self.alpha = alpha

        self.on_vector = None
        self.off_vector = None
        self.obs = None
        self.result = SpectrumObservationList()

    @property
    def npred_source(self):
        """Predicted source `~gammapy.spectrum.CountsSpectrum`.

        Calls :func:`gammapy.spectrum.utils.CountsPredictor`.
        """
        predictor = CountsPredictor(livetime=self.livetime,
                                    aeff=self.aeff,
                                    edisp=self.edisp,
                                    model=self.source_model)
        predictor.run()
        return predictor.npred

    @property
    def npred_background(self):
        """Predicted background (`~gammapy.spectrum.CountsSpectrum`).

        Calls :func:`gammapy.spectrum.utils.CountsPredictor`.
        """
        predictor = CountsPredictor(livetime=self.livetime,
                                    aeff=self.aeff,
                                    edisp=self.edisp,
                                    model=self.background_model)
        predictor.run()
        return predictor.npred

    @property
    def e_reco(self):
        """Reconstructed energy binning."""
        if self.edisp is not None:
            temp = self.edisp.e_reco.bins
        else:
            temp = self.aeff.energy.bins
        return EnergyBounds(temp)

    def run(self, seed):
        """Simulate `~gammapy.spectrum.SpectrumObservationList`.

        The seeds will be set as observation ID.
        Previously produced results will be overwritten.

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
        """Clear all results."""
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
        """Simulate source `~gammapy.spectrum.PHACountsSpectrum`.

        Source counts are added to the on vector.

        Parameters
        ----------
        rand: `~numpy.random.RandomState`
            random state
        """
        on_counts = rand.poisson(self.npred_source.data.data.value)

        on_vector = PHACountsSpectrum(energy_lo=self.e_reco.lower_bounds,
                                      energy_hi=self.e_reco.upper_bounds,
                                      data=on_counts,
                                      backscal=1,
                                      meta=self._get_meta())
        on_vector.livetime = self.livetime
        self.on_vector = on_vector

    def simulate_background_counts(self, rand):
        """Simulate background `~gammapy.spectrum.PHACountsSpectrum`.

        Background counts are added to the on vector.
        Furthermore background counts divided by alpha are added to the off vector.

        TODO: At the moment the source IRFs are used.
        Make it possible to pass dedicated background IRFs.

        Parameters
        ----------
        rand: `~numpy.random.RandomState`
            random state
        """
        bkg_counts = rand.poisson(self.npred_background.data.data.value)
        off_counts = rand.poisson(self.npred_background.data.data.value / self.alpha)

        # Add background to on_vector
        self.on_vector.data.data += bkg_counts * u.ct

        # Create off vector
        off_vector = PHACountsSpectrum(energy_lo=self.e_reco.lower_bounds,
                                       energy_hi=self.e_reco.upper_bounds,
                                       data=off_counts,
                                       backscal=1. / self.alpha,
                                       is_bkg=True,
                                       meta=self._get_meta())
        off_vector.livetime = self.livetime
        self.off_vector = off_vector

    def _get_meta(self):
        """Meta info added to simulated counts spectra."""
        meta = OrderedDict()
        meta['CREATOR'] = self.__class__.__name__
        return meta


class SpectrumEventSampler(object):
    """Simulate events distribution

    The events will be distributed according to a given
    `~gammapy.spectrum.models.SpectralModel`.  For a usage example see
    :gp-extra-notebook:`spectrum_simulation`.

    Parameters
    ----------
    model : `~gammapy.spectrum.model.SpectralModel`
        Spectral model
    n_events : int
        Number of events to draw
    """
    def __init__(self, model, n_events):
        self.model = model
        self.n_events = n_events

        self.events = None

    def draw_events(self, seed):
        """Draw events from model

        The result is stored as ``events`` attribute

        Parameters
        ----------
        seed : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
            see :func:~`gammapy.utils.random.get_random_state`
        """
        random_state = get_random_state(seed)
        if isinstance(self.model, models.PowerLaw):
            events = self._draw_events_powerlaw(random_state,
                                                self.n_events,
                                                self.model)
        else:
            events = self._draw_events_generic(random_state)
        self.events = events

    @staticmethod
    def _draw_events_powerlaw(random_state, n_events, model):
        """Helper function to draw PWL events

        Uses the transformation method described e.g. in Cowan, section 3.2.
        http://www.sherrytowers.com/cowan_statistical_data_analysis.pdf
        """
        # Get n uniformly distributed numbers
        r = random_state.uniform(0, 1, n_events)
        gamma = model.parameters['index'].value

        # transform to PWL distribution
        p = r ** (1 / (1 - gamma))
        ref = model.parameters['reference'].quantity
        events = p * ref
        return events

    def _draw_events_generic(self, random_state):
        """Helper function to draw n events from a generic model

        Uses an accept-reject method with an envelopping power law. Therefore
        the model must be enclosed in a power law. This method is only tested
        for an ECPL.
        """
        # How man events (draw_factor * n_events) are drawn in each turn
        draw_factor = 2
        n_events_turn = draw_factor * self.n_events

        # Generate envelopping PWL
        pars = self.model.parameters
        pars['amplitude'].value = 1
        pwl = models.PowerLaw(index=pars['index'].quantity,
                              amplitude=pars['amplitude'].quantity,
                              reference=pars['reference'].quantity
                              )

        drawn_events = 0
        events = list()
        while drawn_events < self.n_events:
            temp_events = self._draw_events_powerlaw(random_state,
                                                     n_events_turn,
                                                     pwl)

            # Accept reject
            uniform = random_state.uniform(0, 1, n_events_turn)

            flux_pwl = pwl(temp_events) * uniform
            flux_model = self.model(temp_events)

            accepted = np.where(flux_model >= flux_pwl)[0]
            events = np.append(events, temp_events[accepted])
            events._unit = temp_events.unit
            drawn_events = len(events)

        return events[0:self.n_events]
