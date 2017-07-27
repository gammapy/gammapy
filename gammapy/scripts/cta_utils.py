
import numpy as np
import astropy.units as u
from ..spectrum import SpectrumObservation
from ..spectrum.utils import CountsPredictor
from ..spectrum.core import PHACountsSpectrum
from ..utils.random import get_random_state

__all__ = [
    'Target',
    'ObservationParameters',
    'CTAObservationSimulation',
]


class Target(object):
    """Observation target information.

    Parameters
    ----------
    name : `str`
        Name of the source
    model : `~gammapy.spectrum.models.SpectralModel`
        Model of the source
    """

    def __init__(self, name=None, model=None):
        self.name = name
        self.model = model

    def __str__(self):
        """Target report (`str`)."""
        ss = '*** Target parameters ***\n'
        ss += 'Name={}\n'.format(self.name)
        for par in self.model.parameters.parameters:
            ss += '{}={} {}\n'.format(par.name, str(par.value), par.unit)
        return ss

    def from_fermi_lat_catalogue(name):
        raise NotImplementedError


class ObservationParameters(object):
    """Container for observation parameters.

    Parameters
    ----------
    alpha : `~astropy.units.Quantity`
        Normalisation between ON and OFF regions
    livetime :  `~astropy.units.Quantity`
        Observation time
    emin :  `~astropy.units.Quantity`
        Minimal energy for simulation
    emax : `~astropy.units.Quantity`
        Maximal energy for simulation
    """

    def __init__(self, alpha=None, livetime=None,
                 emin=None, emax=None):
        self.alpha = alpha
        self.livetime = livetime
        self.emin = emin
        self.emax = emax

    def __str__(self):
        """Observation summary report (`str`)."""
        ss = '*** Observation parameters summary ***\n'
        ss += 'alpha={} [{}]\n'.format(self.alpha.value, self.alpha.unit)
        ss += 'livetime={} [{}]\n'.format(self.livetime.value,
                                          self.livetime.unit)
        ss += 'emin={} [{}]\n'.format(self.emin.value, self.emin.unit)
        ss += 'emax={} [{}]\n'.format(self.emax.value, self.emax.unit)
        return ss


class CTAObservationSimulation(object):
    """Simulate observation for one IRF and target.

    TODO : Should be merge with `~gammapy.spectrum.SpectrumSimulation`
    """

    @staticmethod
    def simulate_obs(perf, target, obs_param, obs_id=0, random_state='random-seed'):
        """Simulate observation with given parameters.

        Parameters
        ----------
        perf : `~gammapy.scripts.CTAPerf`
            CTA performance
        target : `~gammapy.scripts.Target`
            Source
        obs_param : `~gammapy.scripts.ObservationParameters`
            Observation parameters
        obs_id : `int`, optional
            Observation Id
        random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
            Defines random number generator initialisation.
            Passed to `~gammapy.utils.random.get_random_state`.
        """
        livetime = obs_param.livetime
        alpha = obs_param.alpha.value
        emin = obs_param.emin
        emax = obs_param.emax

        model = target.model

        # Compute expected counts
        reco_energy = perf.bkg.energy
        bkg_rate_values = perf.bkg.data.data * livetime.to('s')
        predicted_counts = CountsPredictor(model=model,
                                           aeff=perf.aeff,
                                           livetime=livetime,
                                           edisp=perf.rmf)
        predicted_counts.run()
        npred = predicted_counts.npred
        # set negative values to zero (interpolation issue)
        idx = np.where(npred.data.data < 0.)
        npred.data.data[idx] = 0

        # Randomise counts
        random_state = get_random_state(random_state)
        on_counts = random_state.poisson(npred.data.data.value)  # excess
        bkg_counts = random_state.poisson(bkg_rate_values.value)  # bkg in ON region
        off_counts = random_state.poisson(bkg_rate_values.value / alpha)  # bkg in OFF region

        on_counts += bkg_counts  # evts in ON region

        on_vector = PHACountsSpectrum(
            data=on_counts,
            backscal=1,
            energy_lo=reco_energy.lo,
            energy_hi=reco_energy.hi,
        )

        on_vector.livetime = livetime
        off_vector = PHACountsSpectrum(energy_lo=reco_energy.lo,
                                       energy_hi=reco_energy.hi,
                                       data=off_counts,
                                       backscal=1. / alpha,
                                       is_bkg=True,
                                       )
        off_vector.livetime = livetime

        obs = SpectrumObservation(on_vector=on_vector,
                                  off_vector=off_vector,
                                  aeff=perf.aeff,
                                  edisp=perf.rmf)
        obs.obs_id = obs_id

        # Set threshold according to the closest energy reco from bkg bins
        idx_min = np.abs(reco_energy.lo - emin).argmin()
        idx_max = np.abs(reco_energy.lo - emax).argmin()
        obs.lo_threshold = reco_energy.lo[idx_min]
        obs.hi_threshold = reco_energy.lo[idx_max]

        return obs

    @staticmethod
    def plot_simu(simu, target):
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
                                       figsize=(10, 5))

        # Spectrum plot
        energy_range = [0.01 * u.TeV, 100 * u.TeV]
        target.model.plot(ax=ax1, energy_range=energy_range,
                          label='Model')
        plt.text(0.55, 0.65, target.__str__(),
                 style='italic', transform=ax1.transAxes, fontsize=7,
                 bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10})
        ax1.set_xlim([energy_range[0].value, energy_range[1].value])
        ax1.set_ylim(1.e-17, 1.e-5)
        ax1.grid(which='both')
        ax1.legend(loc=0)

        # Counts plot
        on_off = simu.on_vector.data.data.value
        off = 1. / simu.off_vector.backscal * simu.off_vector.data.data.value
        excess = on_off - off
        bins = simu.on_vector.energy.lo.value
        x = simu.on_vector.energy.nodes.value
        ax2.hist(x, bins=bins, weights=on_off,
                 facecolor='blue', alpha=1, label='ON')
        ax2.hist(x, bins=bins, weights=off,
                 facecolor='green', alpha=1, label='OFF')
        ax2.hist(x, bins=bins, weights=excess,
                 facecolor='red', alpha=1, label='EXCESS')
        ax2.legend(loc='best')
        ax2.set_xscale('log')
        ax2.set_xlabel('Energy [TeV]')
        ax2.set_ylabel('Expected counts')
        ax2.set_xlim([energy_range[0].value, energy_range[1].value])
        ax2.set_ylim([0.0001, on_off.max() * (1 + 0.05)])
        ax2.vlines(simu.lo_threshold.value, 0, 1.1 * on_off.max(),
                   linestyles='dashed')
        ax2.grid(which='both')
        plt.text(0.55, 0.05, simu.__str__(),
                 style='italic', transform=ax2.transAxes, fontsize=7,
                 bbox={'facecolor': 'white', 'alpha': 1, 'pad': 10})
        plt.tight_layout()
