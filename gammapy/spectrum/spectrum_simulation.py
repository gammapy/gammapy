# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals

from gammapy.utils.random import get_random_state 
from gammapy.spectrum import (
    calculate_predicted_counts,
    PHACountsSpectrum,
    SpectrumObservation,
)

import astropy.units as u

class SpectrumSimulation(object):
    """Simulate `~gammapy.spectrum.SpectrumObservation`
    
    Parameters
    ----------
    aeff : `~gammapy.irf.EffectiveAreaTable`
        Effective Area
    edisp : `~gammapy.irf.EnergyDispersion`,
        Energy Dispserions
    model : `~gammapy.spectrum.models.SpectralModel`
        Source model
    livetime : `~astropy.units.Quantity`
        Livetime
        
    Examples
    --------
    Simulate a spectrum given a model and simulated instrument response functions
     
    .. code-block:: python

        import numpy as np
        import astropy.units as u
        from gammapy.irf import EnergyDispersion, EffectiveArea
        from gammapy.spectrum import models, SpectrumSimulation
        
        e_true = np.logspace(-2, 2.5, 109) * u.TeV
        e_reco = np.logspace(-2,2, 79) * u.TeV

        edisp = EnergyDispersion.from_gauss(
            e_true=e_true,
            e_reco=e_reco,
            sigma=0.2
            )

        aeff = EffectiveAreaTable.from_parametrization(
            energy=e_true)

        model = models.PowerLaw(
            index = 2.3 * u.Unit(''),
            amplitude = 2.5 * 1e-12 * u.Unit('cm-2 s-1 TeV-1'),
            reference = 1 * u.TeV
            )
    
        sim = SpectrumSimulation(
            aeff=aeff,
            edisp=edisp,
            model=model,
            livetime=4*u.h)
    
        obs = sim.simulate_obs(seed=23)
    """
    def __init__(self, aeff, edisp, model, livetime):
        self.aeff = aeff
        self.edisp = edisp
        self.model = model
        self._livetime = livetime
        self._npred = None

    @property
    def livetime(self):
        """Livetime"""
        # This is a property since changing the livetime makes npred invalid
        return self._livetime

    @property
    def npred(self):
        """Prediced source counts"""
        if self._npred is None:
            self._npred = calculate_predicted_counts(livetime=self.livetime,
                                                     aeff=self.aeff,
                                                     edisp=self.edisp,
                                                     model=self.model)
        return self._npred

    def simulate_obs(self, obs_id=1, seed='random-seed', lo_threshold=None,
                     hi_threshold=None):
        """Simulate `~gammapy.spectrum.SpectrumObservation`
        
        Parameters
        ----------
        obs_id : int
            Observation id for simulated obs
        seed : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
            see :func:~`gammapy.utils.random.get_random_state`
        lo_threshold : `~astropy.units.Quantity`, optional
            Low energy threshold, default: 10% of the maximum effective area
        hi_threshold : `~astropy.units.Quantity`, optional
            High energy threshold, default: 50 TeV 
        """
        lo_threshold = lo_threshold or self.aeff.find_energy(
            0.1 * self.aeff.max_area)
        hi_threshold = hi_threshold or 50 * u.TeV

        rand = get_random_state(seed)
        on_counts = rand.poisson(self.npred.data)

        counts_kwargs = dict(energy=self.npred.energy,
                             exposure=self.livetime,
                             obs_id = obs_id,
                             creator = self.__class__.__name__,
                             lo_threshold=lo_threshold,
                             hi_threshold=hi_threshold)

        on_vector = PHACountsSpectrum(data=on_counts,
                                      backscal = 1,
                                      **counts_kwargs)

        obs = SpectrumObservation(on_vector = on_vector,
                                  aeff = self.aeff,
                                  edisp = self.edisp)
        return obs
