# Licensed under a 3 - clause BSD style license - see LICENSE.rst
# my code to include background


from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
import logging
import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord, Angle

from regions import CircleSkyRegion
from ..utils.random import get_random_state
from ..utils.energy import EnergyBounds
from .utils import CountsPredictor
from .core import PHACountsSpectrum
from .observation import SpectrumObservation, SpectrumObservationList
from ..background.reflected import ReflectedRegionsBackgroundEstimator
from ..image import SkyImage
from ..irf import EnergyDispersion, EnergyDispersion2D, EffectiveAreaTable, EffectiveAreaTable2D

__all__ = [
    'SimulationRealBkg'
]

log = logging.getLogger(__name__)

class SimulationRealBkg(object):
    """Simulate `~gammapy.spectrum.SpectrumObservation` with a real background observation.

    Call the ReflectRegionsBackgroundEstimator to estimate background counts/spectra.

    IMPORTANT: Works only for point like IRFs (or very extended sources)

    Parameters
    ----------
    livetime : `~astropy.units.Quantity`, optional [if not specified, takes the observation duration
        Livetime
    source_model : `~gammapy.spectrum.models.SpectralModel`
        Source model
    background_model : `~gammapy.spectrum.models.SpectralModel`, optional
        Background model
    alpha : float, optional
        Exposure ratio between source and background
    obsrun: a real observation run to extract the background from.
    """

    def __init__(self, source_model, obsrun, obspos, livetime=None):
        
        self.source_model = source_model # the model used during simulation
        self.obsrun=obsrun # the observation run
        self.obslist=[obsrun,] #patchy 
        self.obspos=obspos #the source observed
       
        self.offset = SkyCoord.separation(obsrun.pointing_radec,obspos) #calculate the offset

        self.aeff = obsrun.aeff.to_effective_area_table(offset=self.offset)
        self.edisp = obsrun.edisp.to_energy_dispersion(offset=self.offset)
        
        if livetime == None:
            self.src_livetime = obsrun.observation_live_time_duration
        else:
            self.src_livetime = livetime
        self.bkg_livetime = obsrun.observation_live_time_duration
        
        self.on_vector = None
        self.off_vector = None
        self.obs = None
        self.result = SpectrumObservationList()

    @property
    def npred_source(self):
        """Predicted source `~gammapy.spectrum.CountsSpectrum`.

        Calls :func:`gammapy.spectrum.utils.CountsPredictor`.
        Same function as before
        """

        predictor = CountsPredictor(livetime=self.src_livetime,
                                    aeff=self.aeff,
                                    edisp=self.edisp,
                                    model=self.source_model)
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
        obs = SpectrumObservation(on_vector=self.on_vector,
                                  off_vector=self.off_vector,
                                  aeff=self.aeff,
                                  edisp=self.edisp)
        self.simulate_background_counts()
        obs.off_vector=self.off_vector
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
        on_vector.livetime = self.src_livetime
        self.on_vector = on_vector

    def estimate_reflected(self, EXCLUSION_FILE, size):

        #just extracts the reflected background
        on_size=0.11 * u.deg #0.11 for point source cuts...
        allsky_mask = SkyImage.read(EXCLUSION_FILE)
        exclusion_mask = allsky_mask.cutout(position=self.obspos,size=size)
        on_region=CircleSkyRegion(self.obspos,on_size)         
        background_estimator = ReflectedRegionsBackgroundEstimator(obs_list=self.obslist, on_region=on_region, exclusion_mask = exclusion_mask)

        background_estimator.run()
        return background_estimator.result[0]



    def simulate_background_counts(self):

        bkg_res=self.estimate_reflected(EXCLUSION_FILE ='$GAMMAPY_EXTRA/datasets/exclusion_masks/tevcat_exclusion.fits',size=Angle('6 deg'))
        a_off=bkg_res.a_off
        a_on = bkg_res.a_on
        nOFF= len(bkg_res.off_events.table)
        alpha=float(a_on)/float(a_off)
        nbkg=alpha*nOFF # number of background events in on region
        nbkg = np.random.poisson(nbkg)
        idxON=np.random.choice(np.arange(nOFF),nbkg,replace=False)

        bkg_ev=bkg_res.off_events.table[idxON] #background events in on region
        bkg_hist,edge=np.histogram(bkg_ev["ENERGY"],self.on_vector.energy.bins)
        
        # Add background to on_vector
        self.on_vector.data.data += bkg_hist * u.ct

        # Create off vector
        off_events=bkg_res.off_events.table
        off_events.remove_rows(idxON)
        alpha_new=float(a_on)/float(a_off - 1)

        off_counts,edge=np.histogram(off_events["ENERGY"],self.on_vector.energy.bins)
        off_vector = PHACountsSpectrum(energy_lo=self.on_vector.energy.lo,
                                energy_hi=self.on_vector.energy.hi,
                                data=off_counts,
                                backscal=1. / alpha_new, is_bkg=True, meta=self._get_meta())

        self.off_vector = off_vector


    def _get_meta(self):
        """Meta info added to simulated counts spectra."""
        meta = OrderedDict()
        meta['CREATOR'] = self.__class__.__name__
        return meta

