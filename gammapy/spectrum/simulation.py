# Licensed under a 3 - clause BSD style license - see LICENSE.rst
import logging
from ..utils.random import get_random_state
from .utils import SpectrumEvaluator
from .core import CountsSpectrum
from .dataset import SpectrumDatasetOnOff

__all__ = ["SpectrumSimulation"]

log = logging.getLogger(__name__)


class SpectrumSimulation:
    """Simulate `~gammapy.spectrum.SpectrumObservation`.

    For a usage example see :gp-notebook:`spectrum_simulation`

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
        self.result = []

    @property
    def npred_source(self):
        """Predicted source `~gammapy.spectrum.CountsSpectrum`.

        Calls :func:`gammapy.spectrum.utils.SpectrumEvaluator`.
        """
        predictor = SpectrumEvaluator(
            livetime=self.livetime,
            aeff=self.aeff,
            edisp=self.edisp,
            e_true=self.e_true,
            model=self.source_model,
        )
        return predictor.compute_npred()

    @property
    def npred_background(self):
        """Predicted background (`~gammapy.spectrum.CountsSpectrum`).

        Calls :func:`gammapy.spectrum.utils.SpectrumEvaluator`.
        """
        predictor = SpectrumEvaluator(
            livetime=self.livetime,
            aeff=self.aeff,
            edisp=self.edisp,
            e_true=self.e_true,
            model=self.background_model,
        )
        return predictor.compute_npred()

    @property
    def e_reco(self):
        """Reconstructed energy binning."""
        if self.edisp is not None:
            temp = self.edisp.e_reco.edges
        else:
            if self.aeff is not None:
                temp = self.aeff.energy.edges
            else:
                temp = self.e_true
        return temp

    def run(self, seed):
        """Simulate list of `~gammapy.spectrum.SpectrumDatasetOnOff`.

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
        self.result = []
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
            backscale_off = 1 / self.alpha
        else:
            backscale_off = None

        obs = SpectrumDatasetOnOff(
            counts=self.on_vector,
            counts_off=self.off_vector,
            aeff=self.aeff,
            edisp=self.edisp,
            livetime=self.livetime,
            backscale=1,
            backscale_off=backscale_off,
            obs_id=obs_id,
        )
        self.obs = obs

    def simulate_source_counts(self, rand):
        """Simulate source `~gammapy.spectrum.CountsSpectrum`.

        Source counts are added to the on vector.

        Parameters
        ----------
        rand : `~numpy.random.RandomState`
            random state
        """
        on_counts = rand.poisson(self.npred_source.data)

        on_vector = CountsSpectrum(
            energy_lo=self.e_reco[:-1], energy_hi=self.e_reco[1:], data=on_counts
        )
        on_vector.livetime = self.livetime
        self.on_vector = on_vector

    def simulate_background_counts(self, rand):
        """Simulate background `~gammapy.spectrum.CountsSpectrum`.

        Background counts are added to the on vector.
        Furthermore background counts divided by alpha are added to the off vector.

        TODO: At the moment the source IRFs are used.
        Make it possible to pass dedicated background IRFs.

        Parameters
        ----------
        rand : `~numpy.random.RandomState`
            random state
        """
        bkg_counts = rand.poisson(self.npred_background.data)
        off_counts = rand.poisson(self.npred_background.data / self.alpha)

        # Add background to on_vector
        self.on_vector.data += bkg_counts

        # Create off vector
        off_vector = CountsSpectrum(
            energy_lo=self.e_reco[:-1], energy_hi=self.e_reco[1:], data=off_counts
        )
        off_vector.livetime = self.livetime
        self.off_vector = off_vector
