# Licensed under a 3 - clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
import logging
from ..utils.random import get_random_state
from ..utils.energy import EnergyBounds
from .utils import CountsPredictor
from .core import PHACountsSpectrum
from .observation import SpectrumObservation, SpectrumObservationList

__all__ = ["SpectrumSimulation"]

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
    aeff : `~gammapy.irf.EffectiveAreaTable`, optional
        Effective Area
    edisp : `~gammapy.irf.EnergyDispersion`, optional
        Energy Dispersion
    e_true : `~astropy.units.Quantity`, optional
        Desired energy axis of the prediced counts vector if no IRFs are given
    background_model : `~gammapy.spectrum.models.SpectralModel`, optional
        Background model
    alpha : float, optional
        Exposure ratio between source and background
    """

    def __init__(
        self,
        livetime,
        source_model,
        aeff=None,
        edisp=None,
        e_true=None,
        background_model=None,
        alpha=None,
    ):
        self.livetime = livetime
        self.source_model = source_model
        self.aeff = aeff
        self.edisp = edisp
        self.e_true = e_true
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
        predictor = CountsPredictor(
            livetime=self.livetime,
            aeff=self.aeff,
            edisp=self.edisp,
            e_true=self.e_true,
            model=self.source_model,
        )
        predictor.run()
        return predictor.npred

    @property
    def npred_background(self):
        """Predicted background (`~gammapy.spectrum.CountsSpectrum`).

        Calls :func:`gammapy.spectrum.utils.CountsPredictor`.
        """
        predictor = CountsPredictor(
            livetime=self.livetime,
            aeff=self.aeff,
            edisp=self.edisp,
            e_true=self.e_true,
            model=self.background_model,
        )
        predictor.run()
        return predictor.npred

    @property
    def e_reco(self):
        """Reconstructed energy binning."""
        if self.edisp is not None:
            temp = self.edisp.e_reco.bins
        else:
            if self.aeff is not None:
                temp = self.aeff.energy.bins
            else:
                temp = self.e_true
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

    def simulate_obs(self, obs_id, seed="random-seed"):
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
        obs = SpectrumObservation(
            on_vector=self.on_vector,
            off_vector=self.off_vector,
            aeff=self.aeff,
            edisp=self.edisp,
        )
        obs.obs_id = obs_id
        self.obs = obs

    def simulate_source_counts(self, rand):
        """Simulate source `~gammapy.spectrum.PHACountsSpectrum`.

        Source counts are added to the on vector.

        Parameters
        ----------
        rand : `~numpy.random.RandomState`
            random state
        """
        on_counts = rand.poisson(self.npred_source.data.data.value)

        on_vector = PHACountsSpectrum(
            energy_lo=self.e_reco.lower_bounds,
            energy_hi=self.e_reco.upper_bounds,
            data=on_counts,
            backscal=1,
            meta=self._get_meta(),
        )
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
        rand : `~numpy.random.RandomState`
            random state
        """
        bkg_counts = rand.poisson(self.npred_background.data.data.value)
        off_counts = rand.poisson(self.npred_background.data.data.value / self.alpha)

        # Add background to on_vector
        self.on_vector.data.data += bkg_counts

        # Create off vector
        off_vector = PHACountsSpectrum(
            energy_lo=self.e_reco.lower_bounds,
            energy_hi=self.e_reco.upper_bounds,
            data=off_counts,
            backscal=1. / self.alpha,
            is_bkg=True,
            meta=self._get_meta(),
        )
        off_vector.livetime = self.livetime
        self.off_vector = off_vector

    def _get_meta(self):
        return OrderedDict([("CREATOR", self.__class__.__name__)])
